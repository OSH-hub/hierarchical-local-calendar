# Hierarchical local calendar for crop disease warning

Code, aggregate tables and figures for a county-level disease warning model learned from
routine surveillance, and for the verification procedure used to test it against the
national operational warning (apple Marssonina blotch, apple bitter rot, rice leaf blast; Korea).

Every number and figure of the article is produced by the scripts in `src/` and stored in
`tables/` and `figures/`. The figures are drawn from the tables only.

## Contents

| Path | Purpose |
|---|---|
| `src/local_calendar.py` | The model: national, province and county calendars with shrinkage |
| `src/verification.py` | Verification units, contingency scores, cluster bootstrap, year permutation, cost-loss value |
| `src/reference.py` | Seasonal reference forecast, weighted ROC area |
| `src/loader.py` | Reads the input files and checks their SHA-256 checksums |
| `src/run_data_summaries.py` | Descriptions of the input data (tables D01-D04) |
| `src/run_reanalysis.py` | Verification of the operational warning (tables R01-R05, R10) |
| `src/run_controls.py` | Timing controls and their validation, data flow (tables R06-R09) |
| `src/run_local_model.py` | Evaluation of the local calendar, learning curve, new-county adaptation (tables M01-M03) |
| `src/run_competitors.py` | Comparison with five competing forecasting models (table M15) |
| `src/run_local_checks.py` | Sensitivity, permutation test, value, county table, ROC points, updating (tables M04-M14) |
| `src/render_figures.py` | Figures 1-19 of the article and the year-permutation figure for the operational warning |
| `src/make_article_tables.py` | Tables 4 and 5 of the article and the rows added to Table 3 (tables A01, A02) |
| `src/run_all.py` | Runs all of the above in order |
| `examples/` | Synthetic data generator and an end-to-end example |
| `tests/` | Unit tests and an end-to-end test on synthetic data |
| `tables/`, `figures/` | Released outputs; figure file names carry the figure numbers of the article. `figures/year_permutation_operational_warning.*` is the year-permutation control applied to the operational warning (Section 3.4 of the article) |

## Requirements

Python 3.11 with the pinned packages:

```
pip install -r requirements.txt
```

The analysis was run with Python 3.11.9 on Windows 11 (64-bit).

## Quick start without the public data

```
python examples/make_synthetic_data.py example_data
python examples/run_example.py example_data
pytest -q
python src/render_figures.py
```

The example prints, for two synthetic diseases, the ROC area of the synthetic operational
warning, the national calendar and the county calendar. The ten tests finish in a few
seconds. The last command redraws all figures from the released tables;
`python src/make_article_tables.py` rebuilds Tables 4 and 5 of the article from them.

## Reproducing the tables

The inputs are public records of the Rural Development Administration (NCPMS). They are not
redistributed here. Place them under a project directory as

```
data/pairing/pairing_daily.csv                  daily risk level per apple county
data/pairing/pairing_daily_rice.csv             daily risk level per rice county
data/ncpms_api/apple_surveillance_sigungu.csv   apple surveillance records
data/ncpms_api/rice_surveillance_sigungu.csv    rice surveillance records
data/agri_weather/valid_cells.csv               grid-cell coordinates of the apple counties
data/agri_weather/valid_cells_rice.csv          grid-cell coordinates of the rice counties
```

and run

```
python src/run_all.py --project <project directory>
```

The loader stops with an error if a risk or surveillance file differs from the checksums of
the files used in the article. The run takes about five minutes on a desktop computer. All
random draws use the seed 20261026, and two independent runs gave identical tables.
`SHA256SUMS.txt` lists the checksums of the released tables as stored in this repository
(LF line endings); check them with `sha256sum -c SHA256SUMS.txt`.

## Tables

The column "Article" names the section, table or figure of the article that each table supports.
Tables marked "repository only" document the verification of the operational warning and the
checks behind the article; the article does not show them.

| Table | Content | Article |
|---|---|---|
| D01 | Centre of the grid cells of each county | Figures 2, 18 |
| D02 | Valid surveys and day of first detection per county-year | Figures 2, 4, 5 |
| D03 | Share of counties at or above each risk threshold by day and year | Figure 6 |
| D04 | Day of year of every valid survey and whether it was positive | Figure 5 |
| R01 | Contingency scores of the operational warning and the fixed calendars, with intervals | repository only |
| R02 | ROC areas of the operational warning and the seasonal reference | repository only |
| R03 | Paired differences between the operational warning and the fixed calendars | repository only |
| R04 | Year-permutation control of the operational warning | Section 3.4 (p values); `figures/year_permutation_operational_warning` |
| R05a, R05b | Cost-loss value of the operational warning and the fixed calendars | repository only |
| R06 | Season gate fitted out of sample | repository only |
| R07 | Timing controls on the observed archive | repository only |
| R08 | Timing controls on archives with no link to the outcome | Section 3.4, Figure 19 |
| R09 | Records retained at each step | Table 1 |
| R10 | Year-permutation null distribution | `figures/year_permutation_operational_warning` |
| M01 | Scores of the national, province and county calendars under three evaluation designs | Table 3, Table 5 (Brier skill), Figures 7, 8 |
| M02 | Learning curve | Table 4, Figure 12 |
| M03 | Adaptation of a county new to the model | Table 4, Figure 13 |
| M04 | Sensitivity to kernel bandwidth and prior strength | Section 3.1, Figure 11 |
| M05 | County-label permutation test | Table 3 (permutation row), Figure 9 |
| M06 | Scores among county-years surveyed on the usual schedule | repository only |
| M07 | Cost-loss value of the calendars | Table 5, Figure 16 |
| M08 | County calendar table: timing shift, height ratio and period of elevated risk per county | Section 3.3, Table 5 (summary), Figures 3, 17, 18 |
| M09 | Daily probability curves of the national and county calendars | Figures 3, 17 |
| M10 | ROC area of each forecast in each test year | Table 3 (bitter rot, yearly values), Figures 7, 10 |
| M11 | Reliability of the predicted probabilities | Figure 15 |
| M12 | Points of the ROC curves | Figure 7 |
| M13 | County-label permutation null distribution | Figure 9 |
| M14 | Updated and frozen calendars on the same test years | Table 4, Figure 14 |
| M15 | County calendar against five competing models: unpooled county calendar, persistence, mixed-effects logistic regression, random forest, gradient boosting | Table 3 (competing models), Section 3.1 |
| A01 | Table 4 of the article as printed (data efficiency), built from M02, M03 and M14 | Table 4 |
| A02 | Table 5 of the article as printed (practical value), built from M01, M07 and M08 | Table 5 |

The analysis settings of Table 2 of the article are the defaults of the scripts in `src/`.

## Licence

Code: MIT. Tables and figures: CC BY 4.0. The input records remain subject to the terms of
the Rural Development Administration.
