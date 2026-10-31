#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preparation_donnees.py — Préparation du jeu de données du TP « chaleur et performance en course à pied ».

Données d'entrée : resultats.csv (scraping de résultats internationaux, piste et route, 2023-2026), une ligne
par performance, colonnes sexe, annee_naissance, epreuve, meeting, ville, pays (code CIO), date (AAAA-MM-JJ),
heure (HH:MM locale), perf (s), meilleure_perf_saison (s, AVANT la course), record_personnel (s, AVANT la course).

Chaîne complète :
  1. Nettoyage        : types, épreuves normalisées (distance, piste/route, groupe), doublons exacts, contrôle
                        de cohérence de la MPS, âge, identifiants de course, de meeting, de parcours et de lieu.
  2. Géocodage        : hors ligne, base GeoNames « cities1000 » (paquet PyPI reverse_geocoder), avec table de
                        correspondance CIO -> ISO, états américains, alias documentés pour les noms locaux ou les
                        petites communes absentes de la base ; fuseau horaire IANA par timezonefinder (hors ligne).
  3. Météo            : API Open-Meteo Historical Weather (ERA5 / ERA5-Land / IFS, « best_match »), une requête par
                        (lieu, date), réponses mises en cache.
  4. WBGT             : modèle de Liljegren et al. (2008) (wbgt_liljegren.py), calculé toutes les 5 minutes du
                        départ jusqu'à l'arrivée du dernier coureur, puis MOYENNÉ sur la durée de course de chaque
                        athlète (un marathon dure 2 à 3,5 heures).
  5. Cible            : y_pct = 100 ln(temps / record personnel d'avant la course) ; marge de troncature
                        (les listes de résultats s'arrêtent à un temps limite) ; variables de contexte.
  6. Sorties          : dataset_ml.csv, dictionnaire_variables.md/.csv, lieux_geocodes.csv, rapport_preparation.md.

Usage
-----
    python preparation_donnees.py meteo      --resultats resultats.csv --sortie sortie   # Internet requis
    python preparation_donnees.py construire --resultats resultats.csv --sortie sortie
    python preparation_donnees.py tout       --resultats resultats.csv --sortie sortie

Dépendances : pandas, numpy, requests, reverse_geocoder, timezonefinder ; wbgt_liljegren.py dans le même dossier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wbgt_liljegren import esat, wbgt_liljegren, wbgt_simplifie_acsm  # noqa: E402

# =============================================================================================
# Paramètres
# =============================================================================================
MARGE_MIN_PCT = 5.0        # inclusion : record personnel au moins 5 % plus rapide que le dernier temps listé
Y_MIN, Y_MAX = -15.0, 30.0  # bornes de plausibilité de y (en points de pourcentage logarithmique)
RATIO_MPS_RP_MAX = 1.25    # au-delà, la MPS ne peut pas correspondre à la même épreuve : invalidée
PAS_MINUTES = 5            # pas de calcul du WBGT pendant la course
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES_HORAIRES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "surface_pressure", "pressure_msl",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "cloud_cover", "precipitation",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "shortwave_radiation_instant", "direct_radiation_instant", "diffuse_radiation_instant",
]
VARIABLES_SANS_INSTANT = [v for v in VARIABLES_HORAIRES if not v.endswith("_instant")]

EPREUVES = {  # libellé source -> (distance m, support, libellé normalisé, groupe)
    "800 Metres": (800, "piste", "800m", "demi-fond piste"),
    "1000 Metres": (1000, "piste", "1000m", "demi-fond piste"),
    "1500 Metres": (1500, "piste", "1500m", "demi-fond piste"),
    "Mile": (1609.344, "piste", "mile", "demi-fond piste"),
    "2000 Metres Steeplechase": (2000, "piste", "2000m steeple", "fond piste"),
    "2000 Metres Steeplechase (0.84)": (2000, "piste", "2000m steeple", "fond piste"),
    "3000 Metres": (3000, "piste", "3000m", "fond piste"),
    "3000 Metres Steeplechase": (3000, "piste", "3000m steeple", "fond piste"),
    "5000 Metres": (5000, "piste", "5000m", "fond piste"),
    "10,000 Metres": (10000, "piste", "10000m", "fond piste"),
    "Mile Road": (1609.344, "route", "mile route", "route courte"),
    "5 Kilometres Road": (5000, "route", "5 km route", "route courte"),
    "10 Kilometres Road": (10000, "route", "10 km route", "route courte"),
    "Half Marathon": (21097.5, "route", "semi-marathon", "semi-marathon"),
    "Marathon": (42195, "route", "marathon", "marathon"),
}

IOC_ISO2 = {"ARG": "AR", "AUS": "AU", "BEL": "BE", "CAN": "CA", "CHN": "CN", "COL": "CO", "CZE": "CZ", "DEN": "DK",
            "ESP": "ES", "EST": "EE", "FIN": "FI", "FRA": "FR", "GAB": "GA", "GBR": "GB", "GER": "DE", "HKG": "HK",
            "INA": "ID", "IND": "IN", "ITA": "IT", "JPN": "JP", "KOR": "KR", "LAT": "LV", "MAR": "MA", "MEX": "MX",
            "NED": "NL", "NGR": "NG", "PER": "PE", "POL": "PL", "POR": "PT", "QAT": "QA", "RSA": "ZA", "RWA": "RW",
            "SRB": "RS", "SWE": "SE", "THA": "TH", "USA": "US"}
ETATS_US = {"MA": "Massachusetts", "GA": "Georgia", "CO": "Colorado", "IL": "Illinois", "MN": "Minnesota",
            "TX": "Texas", "PA": "Pennsylvania", "OR": "Oregon", "NY": "New York", "ME": "Maine"}
# (ville source, pays CIO) -> (nom GeoNames, admin1 exigé ou None, approximation 0/1, justification)
ALIAS = {
    ("Bengaluru", "IND"): ("Bangalore", None, 0, "Nom GeoNames de Bengaluru."),
    ("Beograd", "SRB"): ("Belgrade", None, 0, "Nom GeoNames de Beograd."),
    ("Ciudad de México", "MEX"): ("Mexico City", None, 0, "Nom GeoNames de Ciudad de México."),
    ("Lisboa", "POR"): ("Lisbon", None, 0, "Nom GeoNames de Lisboa."),
    ("New York, NY", "USA"): ("New York City", None, 0, "Nom GeoNames de New York."),
    ("Osaka", "JPN"): ("Osaka-shi", None, 0, "Nom GeoNames d'Osaka."),
    ("Praha", "CZE"): ("Prague", None, 0, "Nom GeoNames de Praha."),
    ("Warszawa", "POL"): ("Warsaw", None, 0, "Nom GeoNames de Warszawa."),
    ("Ordizia", "ESP"): ("Villafranca de Ordizia", None, 0, "Nom officiel complet d'Ordizia dans GeoNames."),
    ("Heusden-Zolder", "BEL"): ("Heusden", "Flanders", 0, "Heusden (Limbourg), section de la commune de Heusden-Zolder."),
    ("Oordegem", "BEL"): ("Wetteren", None, 1, "Oordegem (commune de Lede) absent de la base : Wetteren, à ~4 km."),
    ("Cape Elizabeth, ME", "USA"): ("South Portland", "Maine", 1, "Cape Elizabeth absent de la base : South Portland, à ~5 km."),
    ("Sunshine Coast", "AUS"): ("Mooloolaba", None, 1, "Région de la Sunshine Coast : Mooloolaba (course à Alexandra Headland, voisine)."),
    ("Bali", "INA"): ("Ubud", None, 1, "Île de Bali : Ubud (régence de Gianyar, où se court le Maybank Marathon)."),
    ("Taiyuan", "CHN"): ("Taiyuan", "Shanxi Sheng", 0, "Homonymes en Chine : capitale du Shanxi."),
    ("Wuxi", "CHN"): ("Wuxi", "Jiangsu Sheng", 0, "Homonymes en Chine : Wuxi (Jiangsu)."),
    ("Yangzhou", "CHN"): ("Yangzhou", "Jiangsu Sheng", 0, "Homonymes en Chine : Yangzhou (Jiangsu)."),
    ("Guiyang", "CHN"): ("Guiyang", "Guizhou Sheng", 0, "Homonymes en Chine : capitale du Guizhou."),
    ("Montreuil", "FRA"): ("Montreuil", "Ile-de-France", 0, "Meeting international de Montreuil (Seine-Saint-Denis)."),
}


def normaliser_nom(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    return " ".join(s.replace("-", " ").replace("’", "'").split())


def hacher(s, n=10):
    return hashlib.sha1(str(s).encode()).hexdigest()[:n]


# =============================================================================================
# 1. Nettoyage
# =============================================================================================
def nettoyer(brut: pd.DataFrame, journal: list) -> pd.DataFrame:
    df = brut.copy()
    for c in df.columns:
        df[c] = df[c].astype("string").str.strip()
    df["ligne_source"] = np.arange(2, len(df) + 2)
    journal.append(("Lignes brutes", len(df), ""))

    # doublons exacts : seulement quand le record personnel est renseigné (ligne identifiante). Sans RP ni année
    # de naissance, deux coureurs différents arrivés dans la même seconde produisent des lignes identiques.
    avec_rp = df["record_personnel"].notna()
    dup = df.drop(columns="ligne_source").duplicated(keep="first") & avec_rp
    journal.append(("Doublons exacts supprimés (lignes avec record personnel)", int(dup.sum()),
                    "lignes " + ", ".join(map(str, df.loc[dup, "ligne_source"].tolist()))))
    n_dup_amb = int((df.drop(columns="ligne_source").duplicated(keep=False) & ~avec_rp).sum())
    journal.append(("Lignes identiques sans record personnel conservées (ex aequo plausibles, hors analyse)", n_dup_amb, ""))
    df = df[~dup].copy()

    inconnues = sorted(set(df["epreuve"]) - set(EPREUVES))
    if inconnues:
        raise ValueError(f"Épreuves inconnues : {inconnues}")
    ep = df["epreuve"].map(EPREUVES)
    df["distance_m"] = [e[0] for e in ep]
    df["support"] = [e[1] for e in ep]
    df["epreuve_norm"] = [e[2] for e in ep]
    df["groupe_epreuve"] = [e[3] for e in ep]
    df["steeple"] = df["epreuve_norm"].str.contains("steeple").astype(int)

    df["temps_s"] = pd.to_numeric(df["perf"], errors="coerce").astype(float)
    df["rp_s"] = pd.to_numeric(df["record_personnel"], errors="coerce").astype(float)
    df["mps_s"] = pd.to_numeric(df["meilleure_perf_saison"], errors="coerce").astype(float)
    incoherente = (df["mps_s"] / df["rp_s"]) > RATIO_MPS_RP_MAX
    journal.append((f"MPS invalidée (MPS / RP > {RATIO_MPS_RP_MAX}, autre épreuve probable)", int(incoherente.sum()),
                    "lignes " + ", ".join(map(str, df.loc[incoherente, "ligne_source"].tolist()))))
    df.loc[incoherente, "mps_s"] = np.nan
    df["vitesse_ms"] = df["distance_m"] / df["temps_s"]
    df["vitesse_rp_ms"] = df["distance_m"] / df["rp_s"]

    df["date_course"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["saison"] = df["date_course"].dt.year
    df["mois"] = df["date_course"].dt.month
    h = pd.to_datetime(df["heure"], format="%H:%M", errors="coerce")
    df["heure_dec"] = h.dt.hour + h.dt.minute / 60.0
    df["heure_type"] = np.where(df["support"] == "piste", "debut_reunion", "depart_course")
    df["annee_naissance"] = pd.to_numeric(df["annee_naissance"], errors="coerce")
    df["age"] = df["saison"] - df["annee_naissance"]

    df["id_meeting"] = (df["meeting"] + "|" + df["ville"] + "|" + df["date"]).map(hacher)
    df["id_course"] = (df["meeting"] + "|" + df["ville"] + "|" + df["date"] + "|" + df["epreuve_norm"] + "|" +
                       df["sexe"]).map(hacher)
    df["id_parcours"] = (df["ville"] + "|" + df["epreuve_norm"]).map(hacher)
    df["id_lieu"] = df["ville"].map(hacher)
    return df


# =============================================================================================
# 2. Géocodage hors ligne
# =============================================================================================
def geocoder(lieux_source, journal):
    import reverse_geocoder
    from timezonefinder import TimezoneFinder
    g = pd.read_csv(Path(reverse_geocoder.__file__).resolve().parent / "rg_cities1000.csv", keep_default_na=False)
    g["cle"] = g["name"].map(normaliser_nom)
    tf = TimezoneFinder()
    lignes = []
    for ville, pays in lieux_source:
        cc = IOC_ISO2[pays]
        nom, admin1, approx, note = ville.split(",")[0], None, 0, ""
        if ", " in ville and pays == "USA":
            admin1 = ETATS_US[ville.split(", ")[1]]
        if (ville, pays) in ALIAS:
            nom, a1, approx, note = ALIAS[(ville, pays)]
            admin1 = a1 or admin1
        cand = g[(g["cle"] == normaliser_nom(nom)) & (g["cc"] == cc)]
        if admin1 is not None:
            cand = cand[cand["admin1"] == admin1]
        if len(cand) != 1:
            raise ValueError(f"Géocodage non résolu : {ville} ({pays}) -> {len(cand)} candidats")
        r = cand.iloc[0]
        tz = tf.timezone_at(lng=float(r["lon"]), lat=float(r["lat"]))
        lignes.append({"ville": ville, "pays": pays, "pays_iso": cc, "nom_geonames": r["name"], "region": r["admin1"],
                       "lat": float(r["lat"]), "lon": float(r["lon"]), "fuseau": tz,
                       "geocodage_approx": approx, "note_geocodage": note})
    lieux = pd.DataFrame(lignes)
    journal.append(("Lieux géocodés (GeoNames cities1000)", len(lieux),
                    f"dont {int(lieux['geocodage_approx'].sum())} par une commune voisine (voir tableau)"))
    return lieux


def ajouter_temps_utc(df, lieux):
    df = df.merge(lieux[["ville", "pays", "lat", "lon", "fuseau", "geocodage_approx"]], on=["ville", "pays"], how="left")
    utc = []
    for d, hdec, tz in zip(df["date_course"], df["heure_dec"], df["fuseau"]):
        hh = int(math.floor(hdec + 1e-9)); mm = int(round((hdec - hh) * 60))
        loc = datetime(d.year, d.month, d.day, hh, mm, tzinfo=ZoneInfo(tz))
        utc.append(loc.astimezone(ZoneInfo("UTC")).replace(tzinfo=None))
    df["datetime_utc"] = pd.to_datetime(pd.Series(utc, index=df.index))
    df["datetime_locale"] = df["date_course"].dt.strftime("%Y-%m-%d") + " " + df["heure"].astype(str)
    return df


# =============================================================================================
# 3. Météo Open-Meteo (archive) : téléchargement avec cache
# =============================================================================================
def _cache_path(dossier, ville, lat, lon, d0, d1):
    h = hacher(f"{ville}|{lat:.5f}|{lon:.5f}|{d0}|{d1}")
    return Path(dossier) / f"{normaliser_nom(ville).replace(' ', '_').replace(',', '')}_{d0}_{h}.json"


def requete_open_meteo(lat, lon, d0, d1, variables, session, essais=5):
    import requests
    params = {"latitude": f"{lat:.5f}", "longitude": f"{lon:.5f}", "start_date": d0, "end_date": d1,
              "hourly": ",".join(variables), "wind_speed_unit": "ms", "timezone": "UTC", "timeformat": "iso8601"}
    r = None
    for k in range(essais):
        try:
            r = session.get(OPEN_METEO_URL, params=params, timeout=60)
        except requests.RequestException:
            if k == essais - 1:
                raise
            time.sleep(2 ** k); continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 400:
            raise ValueError(r.text)
        time.sleep(2 ** k)
    r.raise_for_status()


def telecharger_meteo(df, dossier_cache, journal):
    """Une requête par (lieu, date locale) ; fenêtre [J-1, J+1] en UTC (départs tardifs, marathons, fuseaux)."""
    import requests
    os.makedirs(dossier_cache, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({"User-Agent": "TP-meteo-performance (enseignement)"})
    cles = df[["ville", "lat", "lon", "date_course"]].drop_duplicates()
    lignes, echecs = [], []
    for v, lat, lon, d in cles.itertuples(index=False):
        d0 = (d - timedelta(days=1)).strftime("%Y-%m-%d"); d1 = (d + timedelta(days=1)).strftime("%Y-%m-%d")
        cp = _cache_path(dossier_cache, v, lat, lon, d0, d1)
        if cp.exists():
            js = json.loads(cp.read_text())
        else:
            try:
                try:
                    js = requete_open_meteo(lat, lon, d0, d1, VARIABLES_HORAIRES, sess)
                except ValueError:
                    js = requete_open_meteo(lat, lon, d0, d1, VARIABLES_SANS_INSTANT, sess)
            except Exception as exc:  # noqa: BLE001
                echecs.append(f"{v} {d:%Y-%m-%d} : {exc}")
                continue
            cp.write_text(json.dumps(js))
            time.sleep(0.2)
        hourly = pd.DataFrame(js["hourly"])
        hourly["time_utc"] = pd.to_datetime(hourly.pop("time"))
        hourly["ville"] = v
        hourly["date_course"] = d
        hourly["altitude_m"] = js.get("elevation", np.nan)
        hourly["lat_grille"] = js.get("latitude", np.nan)
        hourly["lon_grille"] = js.get("longitude", np.nan)
        lignes.append(hourly)
    journal.append(("Requêtes météo (lieu x date) réussies", len(lignes),
                    f"{len(echecs)} échec(s)" + (" : " + "; ".join(echecs) if echecs else "")))
    if not lignes:
        raise RuntimeError("Aucune donnée météo récupérée.")
    return pd.concat(lignes, ignore_index=True)


# =============================================================================================
# 4. Météo pendant la course et WBGT moyen sur la durée de course
# =============================================================================================
VARS_INTERP = [("temperature_2m", "t_air"), ("dew_point_2m", "t_rosee"), ("relative_humidity_2m", "hr_openmeteo"),
               ("surface_pressure", "pression_hpa"), ("wind_speed_10m", "vent_10m"), ("cloud_cover", "nebulosite")]


def _serie(t_grille, t_heures, valeurs):
    x = (t_heures - t_heures[0]).total_seconds().to_numpy() / 3600.0
    xg = (t_grille - t_heures[0]).total_seconds().to_numpy() / 3600.0
    v = np.asarray(valeurs, dtype=float)
    ok = ~np.isnan(v)
    if ok.sum() < 2:
        return np.full(len(t_grille), np.nan)
    return np.interp(xg, x[ok], v[ok])


def meteo_pendant_course(df, meteo):
    """Pour chaque départ (lieu, date, heure), météo et WBGT toutes les PAS_MINUTES jusqu'à l'arrivée du dernier
    coureur ; puis, pour chaque performance, moyennes sur [départ, départ + temps]."""
    groupes = {k: g.sort_values("time_utc") for k, g in meteo.groupby(["ville", "date_course"])}
    sorties = []
    cols_moy = ["t_air", "t_rosee", "hr", "hr_openmeteo", "pression_hpa", "vent_10m", "vent_2m", "nebulosite",
                "precipitation_mm_h", "rayonnement_global", "t_globe", "t_nwb", "wbgt", "wbgt_abm"]
    for (v, d, tu), sous in df.groupby(["ville", "date_course", "datetime_utc"]):
        g = groupes.get((v, d))
        if g is None:
            continue
        lat, lon = float(sous["lat"].iloc[0]), float(sous["lon"].iloc[0])
        th = pd.DatetimeIndex(g["time_utc"])
        duree_max = float(np.nanmax(sous["temps_s"])) if sous["temps_s"].notna().any() else 0.0
        n_pas = int(np.ceil(duree_max / 60.0 / PAS_MINUTES)) + 1
        tg = pd.DatetimeIndex([tu + pd.Timedelta(minutes=PAS_MINUTES * k) for k in range(n_pas)])
        s = {nom: _serie(tg, th, g[var]) for var, nom in VARS_INTERP}
        th_moy = th - pd.Timedelta(minutes=30)          # cumuls / moyennes sur l'heure précédente
        s["precipitation_mm_h"] = _serie(tg, th_moy, g["precipitation"])
        if "shortwave_radiation_instant" in g and g["shortwave_radiation_instant"].notna().sum() >= 2:
            s["rayonnement_global"] = _serie(tg, th, g["shortwave_radiation_instant"])
        else:
            s["rayonnement_global"] = _serie(tg, th_moy, g["shortwave_radiation"])
        s["hr"] = np.minimum(100.0, 100.0 * np.array([esat(td + 273.15) / esat(ta + 273.15)
                                                       for ta, td in zip(s["t_air"], s["t_rosee"])]))
        res = [wbgt_liljegren(t.year, t.month, t.day, t.hour, t.minute, lat, lon, max(r, 0.0), p, ta, hr, w,
                              zspeed=10.0, dT=-1.0, urban=True)
               for t, r, p, ta, hr, w in zip(tg, s["rayonnement_global"], s["pression_hpa"], s["t_air"], s["hr"], s["vent_10m"])]
        for k in ["wbgt", "t_globe", "t_nwb", "vent_2m"]:
            s[k] = np.array([r[k] for r in res])
        s["wbgt_abm"] = np.array([wbgt_simplifie_acsm(ta, hr) for ta, hr in zip(s["t_air"], s["hr"])])
        cos_z0 = res[0]["cza"]
        alt = float(g["altitude_m"].iloc[0])
        for idx, t in zip(sous.index, sous["temps_s"]):
            k = int(np.floor(t / 60.0 / PAS_MINUTES)) + 1 if np.isfinite(t) else 1
            rec = {"index": idx, "altitude_m": alt, "cos_zenith_depart": cos_z0, "wbgt_depart": s["wbgt"][0],
                   "wbgt_max_course": float(np.max(s["wbgt"][:k])), "n_pas_meteo": k}
            for c in cols_moy:
                rec[c] = float(np.mean(s[c][:k]))
            sorties.append(rec)
    return pd.DataFrame(sorties).set_index("index")


# =============================================================================================
# 5. Cible, troncature, contexte
# =============================================================================================
def construire_cible(df, journal):
    df = df.copy()
    df["motif_exclusion"] = ""

    def exclure(masque, motif):
        m = masque & (df["motif_exclusion"] == "")
        df.loc[m, "motif_exclusion"] = motif
        journal.append((f"Exclusion : {motif}", int(m.sum()), ""))

    exclure(df["temps_s"].isna(), "temps illisible")
    exclure(df["rp_s"].isna(), "record personnel d'avant course absent (pas de référence)")
    df["y_pct"] = 100.0 * np.log(df["temps_s"] / df["rp_s"])
    exclure((df["y_pct"] < Y_MIN) | (df["y_pct"] > Y_MAX), f"aberrant : y hors de [{Y_MIN:g} ; {Y_MAX:g}] %")
    df["forme_pct"] = 100.0 * np.log(df["mps_s"] / df["rp_s"])
    # troncature : dernier temps listé dans la course (toutes lignes, avec ou sans RP)
    df["t_max_course"] = df.groupby("id_course")["temps_s"].transform("max")
    df["n_listes_course"] = df.groupby("id_course")["temps_s"].transform("size")
    df["marge_troncature_pct"] = 100.0 * np.log(df["t_max_course"] / df["rp_s"])
    valide = df["motif_exclusion"] == ""
    ref = df[valide].groupby(["sexe", "epreuve_norm"])["rp_s"].median()
    df["niveau_pct"] = 100.0 * np.log(df["rp_s"] / pd.Series(list(zip(df["sexe"], df["epreuve_norm"]))).map(ref).values)
    df["duree_ref_min"] = df["rp_s"] / 60.0
    return df


COLONNES = [
    ("ligne_source", "int", "Numéro de ligne dans resultats.csv (en-tête = ligne 1)."),
    ("id_course", "str", "Identifiant de la course : meeting, ville, date, épreuve, sexe."),
    ("id_meeting", "str", "Identifiant du meeting (nom, ville, date)."),
    ("id_parcours", "str", "Identifiant du parcours : ville x épreuve (toutes éditions confondues)."),
    ("id_lieu", "str", "Identifiant du lieu (ville) : unité de regroupement de la validation."),
    ("meeting", "str", "Nom du meeting (brut)."),
    ("ville", "str", "Ville (brute)."),
    ("pays", "str", "Code CIO du pays."),
    ("pays_iso", "str", "Code ISO 3166-1 alpha-2."),
    ("lat", "float", "Latitude (GeoNames)."),
    ("lon", "float", "Longitude (GeoNames, positive vers l'est)."),
    ("fuseau", "str", "Fuseau horaire IANA (timezonefinder)."),
    ("geocodage_approx", "int", "1 si le lieu a été géocodé par une commune voisine (voir rapport)."),
    ("altitude_m", "float", "Altitude (m) du point, renvoyée par Open-Meteo (MNT Copernicus 90 m)."),
    ("date_course", "date", "Date locale de la course."),
    ("saison", "int", "Année."),
    ("mois", "int", "Mois (1-12)."),
    ("heure_dec", "float", "Heure locale de départ (heures décimales)."),
    ("heure_type", "cat", "depart_course (route) ou debut_reunion (piste : l'heure est celle du début de la réunion, la course peut être plus tardive)."),
    ("datetime_locale", "str", "Date et heure locales de départ."),
    ("datetime_utc", "datetime", "Date et heure UTC de départ."),
    ("sexe", "cat", "M ou F."),
    ("annee_naissance", "float", "Année de naissance (souvent manquante)."),
    ("age", "float", "Âge l'année de la course."),
    ("epreuve", "str", "Libellé brut de l'épreuve."),
    ("epreuve_norm", "cat", "Épreuve normalisée (800m, 1500m, 5000m, 10 km route, semi-marathon, marathon...)."),
    ("groupe_epreuve", "cat", "demi-fond piste, fond piste, route courte, semi-marathon, marathon."),
    ("support", "cat", "piste ou route."),
    ("distance_m", "float", "Distance (m)."),
    ("steeple", "int", "1 pour le steeple."),
    ("temps_s", "float", "Temps réalisé (s)."),
    ("vitesse_ms", "float", "Vitesse moyenne réalisée (m/s)."),
    ("rp_s", "float", "Record personnel AVANT la course (s), tel que publié."),
    ("vitesse_rp_ms", "float", "Vitesse du record personnel (m/s)."),
    ("mps_s", "float", "Meilleure performance de la saison AVANT la course (s) ; invalidée si MPS/RP > 1,25."),
    ("forme_pct", "float", "100 ln(MPS / RP) : écart de la forme de la saison au record (manquant si pas de MPS)."),
    ("niveau_pct", "float", "100 ln(RP / RP médian du sexe et de l'épreuve) : < 0 = plus rapide que la médiane."),
    ("duree_ref_min", "float", "Durée de course attendue (RP, en minutes)."),
    ("t_max_course", "float", "Temps du dernier coureur listé dans la course (s) : seuil de troncature observé."),
    ("n_listes_course", "int", "Nombre de coureurs listés dans la course (dépend de la troncature : NE PAS utiliser comme variable explicative)."),
    ("marge_troncature_pct", "float", "100 ln(t_max_course / RP) : ralentissement maximal qui reste observable pour cet athlète."),
    ("t_air", "float", "Température de l'air (°C), moyenne sur la durée de course de l'athlète."),
    ("t_rosee", "float", "Point de rosée (°C), moyenne sur la course."),
    ("hr", "float", "Humidité relative (%), recalculée à partir de T et Td, moyenne sur la course."),
    ("hr_openmeteo", "float", "Humidité relative fournie par Open-Meteo, moyenne sur la course (contrôle)."),
    ("pression_hpa", "float", "Pression de surface (hPa), moyenne sur la course."),
    ("vent_10m", "float", "Vent à 10 m (m/s), moyenne sur la course."),
    ("vent_2m", "float", "Vent extrapolé à 2 m (m/s, Liljegren), moyenne sur la course."),
    ("nebulosite", "float", "Couverture nuageuse (%), moyenne sur la course."),
    ("precipitation_mm_h", "float", "Intensité de précipitation (mm/h), moyenne sur la course."),
    ("rayonnement_global", "float", "Rayonnement solaire global (W/m2), moyenne sur la course."),
    ("cos_zenith_depart", "float", "Cosinus de l'angle zénithal solaire au départ."),
    ("t_globe", "float", "Température de globe (°C), Liljegren, moyenne sur la course."),
    ("t_nwb", "float", "Température humide naturelle (°C), Liljegren, moyenne sur la course."),
    ("wbgt", "float", "WBGT (°C, Liljegren) MOYEN sur la durée de course de l'athlète (pas de 5 min)."),
    ("wbgt_depart", "float", "WBGT au départ (°C)."),
    ("wbgt_max_course", "float", "WBGT maximal pendant la course de l'athlète (°C)."),
    ("wbgt_abm", "float", "Approximation « à l'ombre » du Bureau of Meteorology australien, moyenne (comparaison)."),
    ("y_pct", "float", "CIBLE : 100 ln(temps / record personnel d'avant course). 0 = égale son record ; > 0 = plus lent."),
    ("motif_exclusion", "str", "Vide si la ligne est valide, sinon le motif d'exclusion."),
    ("inclus_analyse", "int", f"1 si : ligne valide, météo disponible et marge de troncature >= {MARGE_MIN_PCT:g} %."),
]


def ecrire_dictionnaire(chemin_md, chemin_csv):
    pd.DataFrame(COLONNES, columns=["variable", "type", "description"]).to_csv(chemin_csv, index=False)
    with open(chemin_md, "w", encoding="utf-8") as f:
        f.write("# Dictionnaire des variables de dataset_ml.csv\n\n")
        f.write("Une ligne = une performance. Encodage UTF-8, séparateur virgule. Les variables météo sont des "
                "moyennes sur la durée de course de l'athlète (du départ à son arrivée).\n\n")
        f.write("| Variable | Type | Description |\n|---|---|---|\n")
        for v, t, d in COLONNES:
            f.write(f"| `{v}` | {t} | {d} |\n")


def ecrire_rapport(chemin, journal, lieux, ds):
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(f"# Rapport de préparation des données\n\nGénéré le {datetime.now():%Y-%m-%d %H:%M}.\n\n")
        f.write("## Journal des opérations\n\n| Opération | Lignes | Détail |\n|---|---|---|\n")
        for op, n, det in journal:
            det = str(det).replace("|", "/").replace("\n", " ")
            f.write(f"| {op} | {n} | {det[:4000]} |\n")
        f.write("\n## Géocodage\n\n")
        f.write(lieux.to_markdown(index=False))
        f.write("\n\n## Échantillon\n\n")
        val = ds["motif_exclusion"].isna() | (ds["motif_exclusion"] == "")
        f.write(f"- Lignes totales : {len(ds)}\n- Lignes valides : {int(val.sum())}\n")
        f.write(f"- Échantillon d'analyse (marge >= {MARGE_MIN_PCT:g} %) : {int(ds['inclus_analyse'].sum())}\n\n")
        if ds["inclus_analyse"].sum() > 0:
            f.write(ds[ds["inclus_analyse"] == 1].groupby("groupe_epreuve").agg(
                performances=("y_pct", "size"), courses=("id_course", "nunique"), lieux=("id_lieu", "nunique")).to_markdown())
        f.write("\n")


def construire(resultats, sortie, meteo_horaire=None, journal=None):
    journal = journal if journal is not None else []
    brut = pd.read_csv(resultats, dtype=str)
    df = nettoyer(brut, journal)
    lieux = geocoder(sorted(set(zip(df["ville"], df["pays"]))), journal)
    df = ajouter_temps_utc(df, lieux)
    df = construire_cible(df, journal)
    if meteo_horaire is not None:
        met = meteo_pendant_course(df, meteo_horaire)
        df = df.join(met)
        journal.append(("Lignes sans météo", int(df["wbgt"].isna().sum()), ""))
    for nom, _, _ in COLONNES:
        if nom not in df.columns:
            df[nom] = np.nan
    valide = df["motif_exclusion"] == ""
    df["inclus_analyse"] = (valide & df["wbgt"].notna() & (df["marge_troncature_pct"] >= MARGE_MIN_PCT)).astype(int)
    journal.append((f"Valides mais marge de troncature < {MARGE_MIN_PCT:g} % (hors analyse)",
                    int((valide & (df["marge_troncature_pct"] < MARGE_MIN_PCT)).sum()), ""))
    ds = df[[c for c, _, _ in COLONNES]].sort_values(["datetime_utc", "id_course", "temps_s"]).reset_index(drop=True)
    sortie = Path(sortie)
    ds.to_csv(sortie / "dataset_ml.csv", index=False, float_format="%.6g")
    lieux.to_csv(sortie / "lieux_geocodes.csv", index=False)
    ecrire_dictionnaire(sortie / "dictionnaire_variables.md", sortie / "dictionnaire_variables.csv")
    ecrire_rapport(sortie / "rapport_preparation.md", journal, lieux, ds)
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etape", choices=["meteo", "construire", "tout"])
    ap.add_argument("--resultats", default="resultats.csv")
    ap.add_argument("--sortie", default=".")
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    sortie = Path(a.sortie); sortie.mkdir(parents=True, exist_ok=True)
    cache = a.cache or str(sortie / "meteo_cache")
    chemin_meteo = sortie / "meteo_horaire.csv"
    journal = []
    if a.etape in ("meteo", "tout"):
        jm = []
        df = nettoyer(pd.read_csv(a.resultats, dtype=str), jm)
        lieux = geocoder(sorted(set(zip(df["ville"], df["pays"]))), jm)
        df = ajouter_temps_utc(df, lieux)
        met = telecharger_meteo(df, cache, jm)
        met.to_csv(chemin_meteo, index=False)
        journal += [e for e in jm if e[0].startswith("Requêtes météo")]
        for op, n, det in journal:
            print(f"{op} : {n} {det}")
        print(f"Météo horaire écrite : {chemin_meteo} ({len(met)} lignes)")
    if a.etape in ("construire", "tout"):
        meteo = pd.read_csv(chemin_meteo, parse_dates=["time_utc", "date_course"]) if chemin_meteo.exists() else None
        if meteo is None:
            print("ATTENTION : meteo_horaire.csv absent, les colonnes météo seront vides.")
        ds = construire(a.resultats, sortie, meteo, journal)
        print(f"dataset_ml.csv : {len(ds)} lignes, dont {int(ds['inclus_analyse'].sum())} dans l'échantillon d'analyse.")


if __name__ == "__main__":
    main()
