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
| `src/run_local_checks.py` | Sensitivity, permutation test, value, county table, ROC points (tables M04-M13) |
| `src/render_figures.py` | Figures 1-20 of the article |
| `src/run_all.py` | Runs all of the above in order |
| `examples/` | Synthetic data generator and an end-to-end example |
| `tests/` | Unit tests and an end-to-end test on synthetic data |
| `tables/`, `figures/` | Released outputs; figure file names carry the figure numbers of the article |

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
seconds. The last command redraws all figures from the released tables.

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

| Table | Content |
|---|---|
| D01 | Centre of the grid cells of each county |
| D02 | Valid surveys and day of first detection per county-year |
| D03 | Share of counties at or above each risk threshold by day and year |
| D04 | Day of year of every valid survey and whether it was positive |
| R01 | Contingency scores of the operational warning and the fixed calendars, with intervals |
| R02 | ROC areas of the operational warning and the seasonal reference |
| R03 | Paired differences between the operational warning and the fixed calendars |
| R04 | Year-permutation control of the operational warning |
| R05a, R05b | Cost-loss value of the operational warning and the fixed calendars |
| R06 | Season gate fitted out of sample |
| R07, R08 | Timing controls on the observed archive and on archives with no link to the outcome |
| R09 | Records retained at each step |
| R10 | Year-permutation null distribution |
| M01 | Scores of the national, province and county calendars under three evaluation designs |
| M02 | Learning curve |
| M03 | Adaptation of a county new to the model |
| M04 | Sensitivity to kernel bandwidth and prior strength |
| M05 | County-label permutation test |
| M06 | Scores among county-years surveyed on the usual schedule |
| M07 | Cost-loss value of the calendars |
| M08 | County calendar table: timing shift, height ratio and period of elevated risk per county |
| M09 | Daily probability curves of the national and county calendars |
| M10 | ROC area of each forecast in each test year |
| M11 | Reliability of the predicted probabilities |
| M12 | Points of the ROC curves |
| M13 | County-label permutation null distribution |

## Licence

Code: MIT. Tables and figures: CC BY 4.0. The input records remain subject to the terms of
the Rural Development Administration.
