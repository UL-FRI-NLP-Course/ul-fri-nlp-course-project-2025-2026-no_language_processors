"""Wrapper to run ADQL queries against Gaia TAP using astroquery."""
import re
import time
import pandas as pd

def run_query(adql: str, retries: int = 3, backoff: float = 15.0, row_limit: int = 50_000) -> pd.DataFrame:
    from astroquery.gaia import Gaia

    Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
    Gaia.ROW_LIMIT = row_limit

    if not re.search(r'\bTOP\s+\d+\b', adql, re.IGNORECASE):
        adql = re.sub(r'\bSELECT\b', f'SELECT TOP {row_limit}', adql, count=1, flags=re.IGNORECASE)

    if not re.search(r'\bWHERE\b', adql, re.IGNORECASE):
        print('⚠️  No WHERE clause — query may time out.')

    print(f'\nQuery sent:\n{adql}\n')

    last_exc = None
    for attempt in range(retries):
        try:
            job = Gaia.launch_job_async(adql, verbose=False)
            result = job.get_results()
            print(f'Got {len(result):,} rows')
            return result.to_pandas()
        except Exception as exc:
            last_exc = exc
            print(f'[attempt {attempt+1}/{retries}] {exc}')
            if '500' in str(exc):
                raise
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                print(f'Waiting {wait:.0f}s …')
                time.sleep(wait)
    raise last_exc
