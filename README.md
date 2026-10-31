# heat-adjusted-pace

**FR** — Quelle allure courir pour fournir le même effort par forte chaleur que par temps idéal ? Ce dépôt rassemble une étude sur 10 398 performances de compétition (800 m au marathon, 100 villes, 36 pays, 2023-2026), un article scientifique, les notebooks de l'analyse, et un calculateur interactif.

**EN** — What pace should you run to produce the same effort in the heat as in ideal conditions? This repository gathers a study of 10,398 race performances (800 m to marathon, 100 cities, 36 countries, 2023-2026), a scientific article, the analysis notebooks, and an interactive calculator.

## Calculateur / Calculator

Ouvrez [`equal_effort_pace_calculator.html`](equal_effort_pace_calculator.html) (bilingue FR/EN, fonctionne dans le navigateur, sans serveur).
Page en ligne / live page : `https://<votre-compte>.github.io/heat-adjusted-pace/equal_effort_pace_calculator.html`

## Contenu / Contents

| | FR | EN |
|---|---|---|
| Article | [PDF](heat_stress_running_performance_fr.pdf) · [LaTeX](heat_stress_running_performance_fr.tex) | [PDF](heat_stress_running_performance_en.pdf) · [LaTeX](heat_stress_running_performance_en.tex) |
| Notebook | [equal_effort_pace_modeling_fr.ipynb](equal_effort_pace_modeling_fr.ipynb) | [equal_effort_pace_modeling_en.ipynb](equal_effort_pace_modeling_en.ipynb) |

- `wbgt_liljegren.py` : modèle WBGT de Liljegren et al. (2008) / WBGT model.
- `preparation_donnees.py` : construction du jeu de données (résultats, géocodage, météo ERA5, WBGT) / data preparation pipeline.
- `lieux_geocodes.csv` : villes géocodées / geocoded cities.
- `processus_torch.py`, `reseau_torch.py` : modules écrits par le notebook (PyTorch dans un processus séparé) / modules written by the notebook.
- `images/` : figures des articles.

## Données / Data

Le jeu de données (`dataset_ml.csv`) **n'est pas distribué** : il provient de résultats de compétition collectés par scraping. Les notebooks sont donc fournis avec leurs sorties, lisibles directement, mais ne peuvent pas être réexécutés sans ce fichier. Le calculateur n'en a pas besoin : il embarque les coefficients du modèle.

The data set (`dataset_ml.csv`) is **not distributed**: it comes from scraped competition results. The notebooks are provided with their outputs and can be read as is, but cannot be re-run without that file. The calculator does not need it: it embeds the model coefficients.

## Avertissement / Disclaimer

Les effets estimés sont des moyennes pour des coureurs de bon niveau. La tolérance individuelle, l'acclimatation et l'hydratation comptent beaucoup ; au-delà de 23 °C de WBGT, la sécurité prime sur le chrono.
The estimated effects are averages for good runners. Individual tolerance, acclimatization and hydration matter a lot; above 23 °C WBGT, safety comes before the race time.

Auteur / Author: G. Lepere
