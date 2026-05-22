Gaia pipeline package

This folder contains a modularised version of the astronomy query notebook. Use the notebook `gaia_pipeline_main.ipynb` as the entrypoint. The package `gaia_pipeline` exposes:

- routed_pipeline(user_query)
- supervised_pipeline(user_query)
- complex_pipeline(user_query)

Notes:
- The LLM parsing is a stub (`parse_query_stub`) — replace with your LLM client integration if needed.
- Queries run against the Gaia TAP service via `astroquery.gaia`.
- The code is simplified for debugging and cluster execution: avoid reading large cache directories.

Requirements:
- astroquery
- pandas

Install with:

pip install astroquery pandas
