"""Modélisation du prix au m² des ventes immobilières en petite couronne parisienne.

Le script télécharge les données DVF et les données carroyées Filosofi,
les enrichit avec des indicateurs communaux de l'API Melodi (Insee),
puis entraîne et évalue un modèle LightGBM.
"""

import os
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from lightgbm import LGBMRegressor
from requests.adapters import HTTPAdapter
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from skore import EstimatorReport
from urllib3.util.retry import Retry

# CONSTANTES -------------------------------------------------

DATA_LOCATION = Path("data")

DEPARTEMENTS_PARIS = ["75", "92", "93", "94"]

SOURCES = {
    "dvf.parquet": (
        "https://minio.lab.sspcloud.fr/projet-formation/"
        "nouvelles-sources/data/geoparquet/dvf.parquet"
    ),
    "filosofi_carreaux_200m_2021.parquet": (
        "https://www.data.gouv.fr/api/1/datasets/r/55432374-a91d-43d0-923d-4514dc3eb951"
    ),
}

FILOSOFI_COLONNES = {
    "idcar_200m": "idcar_200m",
    "ind": "ind",
    "men": "nbre_menages",
    "men_pauv": "nbre_menages_pauvres",
    "men_prop": "nbre_menages_prop",
    "ind_snv": "somme_nv_vie",
    "log_soc": "nbre_logements_sociaux",
    "log_av45": "nbre_logements_avant_1945",
    "men_surf": "surface_totale_logements",
}

JETON_API = "1SuperJetonUltraConfidentiel!!!"

MELODI_URL = "https://api.insee.fr/melodi/data/DS_FILOSOFI_CC"

# Code de la mesure Filosofi dans Melodi -> nom de la colonne créée
INDICATEURS_COMMUNAUX = {
    "MED_SL": "NIVVIE_MEDIAN",
    "PR_MD60": "TAUX_PAUVRETE",
}

NUMERIC_FEATURES = [
    "log_surface", "log_distance_eiffel", "nombre_pieces_principales",
    "nombre_lots", "log_surface_terrain", "ratio_carrez",
    "nbre_menages", "tx_pauvrete", "nv_vie_moyen",
    "part_proprietaires", "part_logements_sociaux",
    "part_logements_avant_1945", "surface_moyenne_logements",
    "NIVVIE_MEDIAN", "TAUX_PAUVRETE",
]
CATEGORICAL_FEATURES = ["code_commune", "type_local"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# LECTURE ET CONTRÔLE DES DONNÉES -----------------------------

def telecharger_sources(sources, data_location):
    """Télécharge les fichiers sources absents du dossier local."""
    data_location.mkdir(exist_ok=True)
    for nom, url in sources.items():
        cible = data_location / nom
        if cible.exists():
            print(f"{nom:42s} déjà présent ({cible.stat().st_size / 1e6:.0f} Mo)")
            continue
        print(f"Téléchargement de {nom} ...")
        reponse = requests.get(url, timeout=120)
        reponse.raise_for_status()
        cible.write_bytes(reponse.content)
        print(f"  -> {cible.stat().st_size / 1e6:.0f} Mo")


def controler_qualite(dvf, variables_obligatoires=("valeur_fonciere", "surface_reelle_bati")):
    """Vérifie la structure de code_commune et les valeurs manquantes."""
    con = duckdb.connect(database=":memory:")
    # DuckDB ne sait pas lire la colonne geometry, on regarde les autres colonnes
    con.register("dvf_attributs", dvf.drop(columns="geometry"))

    # code_commune : 5 caractères qui commencent par le code département
    n_bad = con.sql("""
        SELECT COUNT(*) AS n_bad
        FROM dvf_attributs
        WHERE length(code_commune) <> 5
        OR NOT starts_with(code_commune, code_departement)
    """).fetchall()[0][0]

    if n_bad == 0:
        print("Test 'code_commune' OK: toujours 5 caractères commençant par le département")
    else:
        print(f"Problème dans la colonne code_commune: {n_bad} codes mal formés.")

    for variable in variables_obligatoires:
        n_missing = con.sql(
            f"SELECT COUNT(*) FROM dvf_attributs WHERE {variable} IS NULL"
        ).fetchall()[0][0]
        if n_missing == 0:
            print(f"Pas de valeur manquante pour la variable {variable}")
        else:
            print(f"{n_missing} valeurs manquantes pour la variable {variable}")

    con.close()


# DONNÉES SUPPLÉMENTAIRES -------------------------------------

def ajouter_carreaux_filosofi(dvf, data_location):
    """Associe à chaque vente les données Filosofi de son carreau de 200 m."""
    filo = gpd.read_parquet(
        data_location / "filosofi_carreaux_200m_2021.parquet",
        columns=[*FILOSOFI_COLONNES, "geometry"],
    ).rename(columns=FILOSOFI_COLONNES)

    print(
        f"Reprojection de {dvf.shape[0]} transactions immobilières "
        f"du CRS {dvf.crs.to_epsg()} à {filo.crs.to_epsg()}"
    )
    dvf = dvf.to_crs(filo.crs)
    return dvf.sjoin(filo, how="left", predicate="within")


def creer_session_http():
    """Crée une session HTTP qui réessaie en cas de coupure réseau ou d'erreur serveur."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def recuperer_indicateur_melodi(session, code, value_name, time_period=2023):
    """Récupère un indicateur Filosofi communal depuis l'API Melodi."""
    params = {
        # Dans DVF, Paris est codé par arrondissement (75101 à 75120) : avec le
        # seul niveau COM (Paris = 75056), toutes les ventes parisiennes
        # perdraient leurs variables communales.
        "GEO": ["COM", "ARM"],
        "FILOSOFI_MEASURE": code,
        "TIME_PERIOD": time_period,
        "maxResult": 40000,
    }
    headers = {"Authorization": f"Bearer {JETON_API}"}

    reponse = session.get(MELODI_URL, params=params, headers=headers, timeout=60)
    reponse.raise_for_status()

    return pd.DataFrame(
        {
            "CODGEO": obs["dimensions"]["GEO"].split("-")[-1],
            value_name: obs["measures"]["OBS_VALUE_NIVEAU"].get("value"),
        }
        for obs in reponse.json()["observations"]
    )


def ajouter_donnees_communales(dvf):
    """Ajoute le niveau de vie médian et le taux de pauvreté de la commune."""
    session = creer_session_http()
    for code, value_name in INDICATEURS_COMMUNAUX.items():
        indicateur = recuperer_indicateur_melodi(session, code, value_name)
        dvf = dvf.merge(
            indicateur, left_on="code_commune", right_on="CODGEO", how="left"
        ).drop(columns="CODGEO")
    return dvf


# EXPLORATION ET FEATURE ENGINEERING --------------------------

def statistiques_descriptives(dvf):
    """Affiche un aperçu des données et trace leur distribution."""
    print(dvf.head(5))
    print("Statistiques descriptives ---------------------")
    sns.histplot(data=dvf, x="valeur_fonciere")
    sns.histplot(data=dvf, x="valeur_fonciere", log_scale=True)
    grid = sns.scatterplot(data=dvf, x="surface_reelle_bati", y="valeur_fonciere")
    grid.set(xscale="log", yscale="log")


def filtrer_prix_extremes(dvf):
    """Écarte les prix au m² extrêmes, globalement puis commune par commune."""
    # Centiles inférieur et supérieur : ventes multi-lots, démembrements ou
    # erreurs de saisie DVF. Sans ce filtre le modèle apprend en bonne partie
    # du bruit (R² test ~0.39 contre ~0.61 une fois ces valeurs écartées).
    borne_basse, borne_haute = dvf["prix_m2"].quantile([0.01, 0.99])
    dvf = dvf.loc[dvf["prix_m2"].between(borne_basse, borne_haute)]

    # Les bornes ci-dessus sont tirées par Paris (≈ 21 000 €/m² en haut) : en
    # banlieue, des ventes à 15-20 000 €/m² (erreurs de surface, ventes groupées)
    # passent le filtre et pèsent ~1/3 de l'erreur quadratique hors Paris.
    # On applique donc le même filtre au marché local, commune par commune.
    prix_par_commune = dvf.groupby("code_commune")["prix_m2"]
    return dvf.loc[
        dvf["prix_m2"].between(
            prix_par_commune.transform("quantile", 0.01),
            prix_par_commune.transform("quantile", 0.99),
        )
    ].copy()


def creer_variables(dvf):
    """Construit les variables explicatives et la cible prix_m2."""
    print("Feature engineering")
    # Dans Filosofi, chaque ménage occupe un logement : on peut donc rapporter
    # les comptes de logements au nombre de ménages du carreau.
    for nouvelle, numerateur in {
        "tx_pauvrete": "nbre_menages_pauvres",
        "nv_vie_moyen": "somme_nv_vie",
        "part_proprietaires": "nbre_menages_prop",
        "part_logements_sociaux": "nbre_logements_sociaux",
        "part_logements_avant_1945": "nbre_logements_avant_1945",
        "surface_moyenne_logements": "surface_totale_logements",
    }.items():
        dvf[nouvelle] = dvf[numerateur] / dvf["nbre_menages"]
    dvf["prix_m2"] = dvf["valeur_fonciere"] / dvf["surface_reelle_bati"]

    dvf = filtrer_prix_extremes(dvf)

    dvf["log_surface"] = np.log(dvf["surface_reelle_bati"])

    # Ce qui est vendu en plus du logement lui-même : le prix DVF inclut les
    # dépendances (cave, parking = lots supplémentaires) et le terrain des maisons.
    # Le rapport surface Carrez / surface bâtie signale les ventes où la surface
    # déclarée ne correspond pas au bien (pas de Carrez hors copropriété -> NaN).
    dvf["log_surface_terrain"] = np.log1p(dvf["surface_terrain"].fillna(0))
    dvf["ratio_carrez"] = dvf["lot1_surface_carrez"] / dvf["surface_reelle_bati"]

    # Distance à la tour Eiffel (proxy de centralité, très corrélé au prix en IdF).
    # On calcule la distance en Lambert-93 (mètres) puis on passe en log(1 + km).
    tour_eiffel = gpd.GeoSeries.from_xy(
        [2.2945], [48.8584], crs="EPSG:4326"
    ).to_crs("EPSG:2154").iloc[0]
    distance_km = dvf.geometry.to_crs("EPSG:2154").distance(tour_eiffel) / 1000
    dvf["log_distance_eiffel"] = np.log1p(distance_km)
    return dvf


def afficher_prix_m2_commune(dvf):
    """Affiche les communes au prix moyen au m² le plus bas et le plus élevé."""
    prix_m2_commune = (
        dvf.groupby(["code_commune", "nom_commune"])["prix_m2"].mean().sort_values()
    )
    print("Bottom 5 communes (prix au m2 le plus bas) :")
    print(prix_m2_commune.head(5))
    print("Top 5 communes (prix au m2 le plus élevé) :")
    print(prix_m2_commune.tail(5))


# PIPELINE ET MODÈLE ------------------------------------------

def as_category(df):
    """Convertit les colonnes en dtype 'category' pour que LightGBM les
    reconnaisse automatiquement comme variables catégorielles."""
    return df.astype("category")


def construire_pipeline():
    """Construit le pipeline de préparation des données et le modèle LightGBM."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ("as_category", FunctionTransformer(as_category)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("Preprocessing numerical", numeric_transformer, NUMERIC_FEATURES),
            ("Preprocessing categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    ).set_output(transform="pandas")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=0,
            verbose=-1,
        )),
    ])

    # Le prix au m² est très asymétrique : on apprend sur log(1 + prix) et on
    # repasse en euros à la prédiction. La perte quadratique est alors moins
    # dominée par les transactions les plus chères.
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1
    )


def decouper_et_sauvegarder(dvf_start_data):
    """Découpe les données en apprentissage / test et les sauvegarde en CSV."""
    x = dvf_start_data[FEATURES]
    y = dvf_start_data["prix_m2"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=0
    )
    pd.concat([x_train, y_train], axis=1).to_csv("train.csv")
    pd.concat([x_test, y_test], axis=1).to_csv("test.csv")
    return x_train, x_test, y_train, y_test


def verifier_modalites(x_train, x_test):
    """Vérifie que toutes les modalités du test sont connues du modèle."""
    for variable in CATEGORICAL_FEATURES:
        inconnues = (
            set(x_test[variable].dropna().unique())
            - set(x_train[variable].dropna().unique())
        )
        if inconnues:
            print(f"Modalités de {variable} absentes de l'apprentissage")
        else:
            print(f"Toutes les modalités de {variable} du test sont connues du modèle")


# ÉVALUATION --------------------------------------------------

def afficher_rmse(pipe, x_train, x_test, y_train, y_test):
    """Affiche la RMSE sur l'apprentissage et sur le test."""
    rmse_train = root_mean_squared_error(y_train, pipe.predict(x_train))
    rmse_test = root_mean_squared_error(y_test, pipe.predict(x_test))
    print(f"RMSE sur l'apprentissage : {rmse_train:,.0f} €/m²")
    print(f"RMSE sur le test : {rmse_test:,.0f} €/m²")


def afficher_rmse_par_commune(pipe, x_test, y_test, cog):
    """Affiche les communes où le modèle se trompe le moins et le plus."""
    residuals = x_test[["code_commune"]].copy()
    residuals["y_test"] = y_test
    residuals["pred"] = pipe.predict(x_test)

    rmse_commune = (
        residuals.groupby("code_commune")
        .apply(
            lambda d: root_mean_squared_error(d["y_test"], d["pred"]),
            include_groups=False,
        )
        .rename("rmse")
        .reset_index()
        .merge(cog, on="code_commune")
        .sort_values("rmse")
    )
    print("Bottom 5 communes (RMSE la plus faible) :")
    print(rmse_commune.head(5))
    print("Top 5 communes (RMSE la plus élevée) :")
    print(rmse_commune.tail(5))


def evaluer_avec_skore(pipe, x_train, x_test, y_train, y_test):
    """Produit le rapport skore : métriques, erreurs de prédiction et importances."""
    print("Évaluation skore ---------------------")

    # skore réentraîne une copie du pipeline sur le même découpage
    report = EstimatorReport(
        clone(pipe), X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test
    )
    print(report)

    # Tableau de métriques (RMSE, MAE, R², temps de calcul) sur train et test
    print(report.metrics.summarize(data_source="both").frame())

    # Graphique des erreurs de prédiction (prédit vs réel)
    report.metrics.prediction_error().plot(kind="actual_vs_predicted")
    plt.savefig("prediction_error.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Importance des variables par permutation
    print(report.inspection.permutation_importance(data_source="test").frame())


# SCRIPT PRINCIPAL --------------------------------------------

def main():
    """Enchaîne la préparation des données, l'entraînement et l'évaluation."""
    os.chdir(Path(__file__).resolve().parent)

    print("Lecture des données ------------------------")
    telecharger_sources(SOURCES, DATA_LOCATION)
    dvf = gpd.read_parquet(DATA_LOCATION / "dvf.parquet")
    controler_qualite(dvf)

    dvf = dvf.loc[dvf["valeur_fonciere"] < 1e6].copy()
    dvf["annee"] = pd.to_datetime(dvf["date_mutation"]).dt.year

    dvf = ajouter_carreaux_filosofi(dvf, DATA_LOCATION)
    print("Restriction aux départements de la petite couronne")
    dvf = dvf.loc[dvf["code_departement"].isin(DEPARTEMENTS_PARIS)]
    dvf = ajouter_donnees_communales(dvf)

    statistiques_descriptives(dvf)
    dvf = creer_variables(dvf)
    afficher_prix_m2_commune(dvf)

    # Les valeurs manquantes des variables explicatives (ratio_carrez hors
    # copropriété, quelques carreaux ou communes sans donnée Filosofi) sont
    # imputées dans le pipeline : on n'écarte que les lignes sans cible.
    dvf_start_data = dvf.dropna(subset=["prix_m2"])
    x_train, x_test, y_train, y_test = decouper_et_sauvegarder(dvf_start_data)
    verifier_modalites(x_train, x_test)

    pipe = construire_pipeline()
    pipe.fit(x_train, y_train)

    afficher_rmse(pipe, x_train, x_test, y_train, y_test)
    cog = dvf_start_data[["code_commune", "nom_commune"]].drop_duplicates()
    afficher_rmse_par_commune(pipe, x_test, y_test, cog)
    evaluer_avec_skore(pipe, x_train, x_test, y_train, y_test)

    print("Fin du script !")


if __name__ == "__main__":
    main()
