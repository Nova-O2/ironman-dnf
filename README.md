# Who Drops Out? Non-Completion in IRONMAN® Triathlon — Analysis Code and Results

Analysis code, result files, tables and figures for the study:

**"Who Drops Out? A Population-Level Epidemiology of Non-Completion in IRONMAN® Triathlon"**

Population-level analysis of non-completion (did not finish, DNF) in IRONMAN® and IRONMAN® 70.3 triathlon: 3,586,149 athlete-race records, 2002–2026 (2026 partial), with DNF rates by sex, age group, region and event type, the segment of withdrawal, an interval-censored dropout hazard along the run, temporal trends and a risk-factor model.

The citation will be added upon publication.

## Authors

| Author | Affiliation | ORCID |
|--------|-------------|-------|
| Aldo Seffrin | Nova O2 Sports Science, São José dos Campos, São Paulo, Brazil | [0000-0001-8229-8565](https://orcid.org/0000-0001-8229-8565) |
| Paulo Puccinelli | Centro Universitário São Camilo, São Paulo, Brazil | [0000-0003-1785-753X](https://orcid.org/0000-0003-1785-753X) |
| Pantelis Theodoros Nikolaidis | School of Health and Caring Sciences, University of West Attica, Athens, Greece | [0000-0001-8030-7122](https://orcid.org/0000-0001-8030-7122) |
| Marilia Santos Andrade | Department of Physiology, Federal University of São Paulo (UNIFESP), São Paulo, Brazil | [0000-0002-7004-4565](https://orcid.org/0000-0002-7004-4565) |
| Elias Villiger | Institute of Primary Care, University of Zurich, Zurich, Switzerland | [0000-0001-8371-1390](https://orcid.org/0000-0001-8371-1390) |
| Thomas Rosemann | Institute of Primary Care, University of Zurich, Zurich, Switzerland | [0000-0002-6436-6306](https://orcid.org/0000-0002-6436-6306) |
| Katja Weiss | Institute of Primary Care, University of Zurich, Zurich, Switzerland | [0000-0003-1247-6754](https://orcid.org/0000-0003-1247-6754) |
| Daniel Ferreira | Nova O2 Sports Science, São José dos Campos, São Paulo, Brazil | [0000-0002-8958-9442](https://orcid.org/0000-0002-8958-9442) |
| Beat Knechtle (corresponding author) | Medbase St. Gallen am Vadianplatz, St. Gallen & Institute of Primary Care, University of Zurich, Zurich, Switzerland | [0000-0002-2412-9103](https://orcid.org/0000-0002-2412-9103) |

## Dataset

The study analyses version 4.0.0 of the IRONMAN® triathlon dataset described in the companion data descriptor (*Sci* 2026, 8(9), 265, https://doi.org/10.3390/sci8090265). The de-identified dataset is openly available in Zenodo under a CC BY 4.0 licence: version 4.0.0, https://doi.org/10.5281/zenodo.23421027 (concept DOI 10.5281/zenodo.19284673). It is not redistributed here.

The deposited file omits 93 secondary-source records whose identifiers were redacted at source; rerunning the pipeline on it lowers some record counts by at most 93 and leaves every rate, ratio, hazard and regression estimate unchanged at the precision reported (`notebooks/README.md`, *Reproduction check on the deposited file*; `notebooks/results/deposit_rerun_audit.csv`).

The race-location metadata (`all_subevents.csv`, `race_metadata.csv`) and the quality-assessment outputs of dataset version 4.0.0 are in the companion data repository, https://github.com/Nova-O2/ironman-data (folders `data/collection` and `notebooks/results_v4`).

## Repository structure

```
ironman-dnf/
├── notebooks/                  # Analysis pipeline (Python 3.12+); see notebooks/README.md
│   ├── _common.py              # Data loading, analysed population, course positions, shared helpers
│   ├── _race_locations.py      # Race → host country → continent
│   ├── descriptives.py         # DNF rates by sex, age group, region and event type
│   ├── dropout_point.py        # Segment of withdrawal (distance and split methods, cross-tabulation)
│   ├── dropout_hazard.py       # Interval-censored dropout hazard per km along the run
│   ├── temporal_trends.py      # Annual DNF rates and trend tests, 2002–2026
│   ├── athlete_vs_venue.py     # Athlete origin × race location
│   ├── pro_rank_analysis.py    # Professional athletes
│   ├── dnf_prediction.py       # Logistic regression (risk factors)
│   ├── sensitivity_analyses.py # Sensitivity models S1–S4
│   ├── generate_tables.py      # Tables from the result files
│   ├── generate_figures.py     # Figures 1–5 (TIFF 300 DPI and vector PDF)
│   ├── manuscript_number_audit.py  # Every reported number checked against results/
│   └── results/                # Result files (txt and csv), each with a provenance header
├── figures/                    # Figures 1–5 (TIFF and PDF)
├── manuscript/tables/          # Tables 1–3 and S1–S8 (Markdown, generated)
├── requirements.txt            # Pinned Python dependencies
└── LICENSE                     # MIT
```

## Reproducing the analyses

```bash
pip install -r requirements.txt

export IRONMAN_DNF_DATA_PATH=/abs/path/to/ironman_dataset_deidentified.csv
export IRONMAN_DNF_SUBEVENTS_PATH=/abs/path/to/ironman-data/data/collection/all_subevents.csv
export IRONMAN_DNF_RACE_METADATA_PATH=/abs/path/to/ironman-data/data/collection/race_metadata.csv
# optional: write the outputs elsewhere instead of overwriting results/, figures/ and manuscript/tables/
# export IRONMAN_DNF_RESULTS_DIR=... IRONMAN_DNF_TABLE_DIR=... IRONMAN_DNF_FIGURES_DIR=...

python3 notebooks/descriptives.py
python3 notebooks/dropout_point.py
python3 notebooks/dropout_hazard.py
python3 notebooks/temporal_trends.py
python3 notebooks/athlete_vs_venue.py
python3 notebooks/pro_rank_analysis.py
python3 notebooks/dnf_prediction.py
python3 notebooks/sensitivity_analyses.py
python3 notebooks/generate_tables.py
python3 notebooks/generate_figures.py
python3 notebooks/manuscript_number_audit.py
```

Use absolute paths. The execution order, the environment variables, the analysed population and the methodological decisions are documented in `notebooks/README.md`. The published figures use Liberation Sans (not distributed here); without it the figures are drawn with a fallback font.

## License

Code: MIT — Copyright (c) 2026 Nova O2. The dataset keeps its own licence (CC BY 4.0, Zenodo).
