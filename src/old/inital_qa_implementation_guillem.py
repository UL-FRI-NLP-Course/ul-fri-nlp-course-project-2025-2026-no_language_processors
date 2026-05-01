import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from astroquery.gaia import Gaia

if __name__ == '__main__':
    adql_query = """
    SELECT 
        source_id, ra, dec, parallax, phot_g_mean_mag 
    FROM gaiadr3.gaia_source 
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', 56.75, 24.116, 0.5)
    )
    """

    job = Gaia.launch_job_async(adql_query)
    results = job.get_results()
    print(results)