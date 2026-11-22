# Data access and reproduction

The analysis uses two distinct sources from the Rural Development Administration (RDA), Republic of Korea. They should not be described as one universally redistributable open dataset.

## Official surveillance API

The original collection scripts used `https://ncpms.rda.go.kr/npmsAPI/service` with service codes SVC51 (survey rounds) and SVC53 (county surveillance). The [Korean Public Data Portal entry 15058192](https://www.data.go.kr/data/15058192/openapi.do) lists unrestricted permitted use. The [NCPMS API information page](https://ncpms.rda.go.kr/npms/OpenApiInfo.np) describes registration and API application. Follow its current procedures for an API key; keys are not included in this package. The portal's entry was checked on 26 October 2026.

The frozen apple and rice surveillance inputs are identified by the two `data/ncpms_api/*_surveillance_sigungu.csv` entries in `source_checksums.json`. Analysis is restricted to the years, disease indices and counties specified in the manuscript. All-index raw row counts, which include 2026, are not the analysis denominators.

## Historical map risk series

Risk values were originally retrieved through the public map's `NewSpaceDistributionSearchAjax.np` endpoint. This is separate from the documented surveillance API. The archive contains records returned for past dates, not independently verified as-of forecast issuance. The package does not infer that the surveillance API license covers the map archive. [NCPMS source terms](https://ncpms.rda.go.kr/npms/DatacprprotM.np) govern applicable use; [RDA's general copyright policy](https://www.rda.go.kr/datacprprot.do) distinguishes works for which RDA holds all economic rights.

The two frozen `data/pairing/pairing_daily*.csv` inputs have checksums in `source_checksums.json`. The final rice input is the source-corrected series with SHA-256 `dad24668e0cdd1696c0e2f7057bf4d55da8e83e0f8701664d292d9f072d86fbc`. Recollection from a current service may not yield the same historical values. The archived provider model version and issue timestamp remain unknown.

## What is supplied

This package supplies analysis scripts, aggregate result tables, figures, protocols and source checksums. It excludes original database responses and individual event, episode and day-level outputs. The supplied aggregates permit checks of reported counts, rates, sensitivity contrasts and plots. Full bootstrap and raw-input reproduction require access to the corresponding source inputs.

The author retains the frozen inputs and audit outputs. An editor or reviewer seeking source-level verification can contact the corresponding author of the article to identify the exact inputs and applicable provider access route. Direct source-data inspection or transfer must follow the provider's conditions and be arranged separately. This statement does not promise unrestricted third-party redistribution or an already established reviewer-access service.
