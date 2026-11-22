# disease-warning-verification

Analysis code and aggregate results for the article

> Oh, S. *A Verification Framework for Archived Crop Disease Risk Warnings: Detection, Warning Burden and False Alarms in Korean Apple and Rice Pathosystems*.

The framework verifies archived crop-disease risk warnings against county surveillance records and reports three things together: event detection, warning-day burden relative to fixed seasonal calendars, and the false-alarm ratio under explicit episode-eligibility rules. It is applied to apple Marssonina blotch (AMB), apple anthracnose (ABR) and rice leaf blast (RLB; code `RB_IR` in the tables) in Korea.

## Contents

| Path | Contents |
|---|---|
| `reproducibility/recalculate_main_analysis.py` | Main analysis (produces Data S1–S10) |
| `reproducibility/complete_followup_sensitivity.py` | Complete-follow-up sensitivity (produces Data S11) |
| `reproducibility/render_figures_from_tables.py` | Redraws the four result figures from the aggregate tables |
| `reproducibility/analysis_protocol.md`, `reproducibility/protocol_amendment_source_correction.md` | Analysis protocol extract and source-correction amendment |
| `reproducibility/source_checksums.json` | SHA-256 checksums of the four source inputs |
| `reproducibility/data_access.md` | Source access routes and conditions |
| `tables/DataS01_metrics_support_and_phase.csv` | Data S1: all thresholds, comparators, supports and episode populations |
| `tables/DataS02_paired_calendar_comparison.csv` | Data S2: paired calendar comparisons |
| `tables/DataS03_matched_warning_budget_shifts.csv` | Data S3: equal-budget circular-shift summaries |
| `tables/DataS04_temporal_seasonal_control.csv` | Data S4: earlier-year fitted seasonal controls |
| `tables/DataS05_year_inventory.csv` | Data S5: risk and surveillance availability by year |
| `tables/DataS06_temporal_sensitivity.csv` | Data S6: temporal sensitivity |
| `tables/DataS07_observation_quality_sensitivity.csv` | Data S7: observation-quality sensitivity |
| `tables/DataS08_province_omission.csv` | Data S8: province omission |
| `tables/DataS09_window_gap_sensitivity.csv` | Data S9: window/gap sensitivity |
| `tables/DataS10_source_version_sensitivity.csv` | Data S10: source-version sensitivity |
| `tables/DataS11_complete_followup.csv`, `tables/DataS11_complete_followup_checks.json` | Data S11: complete-follow-up results and checks |
| `figures/` | The four result figures (Figures 2–5 of the article; files `figure1_…` to `figure4_…`) as PNG and PDF |
| `SHA256SUMS.txt` | SHA-256 checksums of the files in this repository |

## Included and excluded

Included: analysis code, aggregate result tables, figure sources, the protocol extract and checksums of the source inputs.

Excluded: the original responses of the provider's services and the derived event, episode and day-level records. The daily risk archive was retrieved from a public map endpoint of the National Crop Pest Management System (NCPMS) of the Rural Development Administration, Republic of Korea, and remains subject to the provider's copyright notice; it is therefore not redistributed here. No access keys or credentials are included.

## Running

Requirements: Python 3.11, NumPy, pandas, Matplotlib.

Redraw the figures from the aggregate tables (no source data needed):

    python reproducibility/render_figures_from_tables.py

Recalculate the analysis from the source inputs (requires the four inputs listed in `reproducibility/source_checksums.json`, arranged in a project directory):

    python reproducibility/recalculate_main_analysis.py --project PATH_TO_PROJECT [--rice-source PATH_TO_RICE_DAILY_CSV]
    python reproducibility/complete_followup_sensitivity.py --project PATH_TO_PROJECT --episodes results/episodes.csv

The main script checks the SHA-256 of all four inputs before calculation.

## Data sources and conditions of use

- Surveillance records: NCPMS OpenAPI (https://ncpms.rda.go.kr/npms/OpenApiInfo.np), listed on the Korean Public Data Portal as dataset 15058192 (https://www.data.go.kr/data/15058192/openapi.do).
- Daily risk series: public forecast-map page of NCPMS (https://ncpms.rda.go.kr/npms/NewSpaceDistributionSearchR.np), subject to the provider's data-use and copyright notice (https://ncpms.rda.go.kr/npms/DatacprprotM.np).

See `reproducibility/data_access.md` for details.

## License and citation

Code and aggregate tables are released under the MIT License (see `LICENSE`). Please cite the article; citation metadata are in `CITATION.cff`.
