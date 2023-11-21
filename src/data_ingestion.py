import argparse
import datetime
import os
import re
import threading
import time
from pathlib import Path

import pandas as pd

from data_monitoring import get_logger
from utils import perform_get_request, xml_to_load_dataframe, xml_to_gen_data, get_chunks_between_dates

URL_API = 'https://web-api.tp.entsoe.eu/api'
SECURITY_TOKEN = '1d9cd4bd-f8aa-476c-8cc1-3442dc91506d'
PROJECT_PATH = Path(__file__).parent.resolve() / '../'

LOGGER = get_logger("data_ingestion")


def get_load_data_from_entsoe(regions, periodStart='202302240000', periodEnd='202303240000', output_path='./data'):
    # General parameters for the API
    # Refer to https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html#_documenttype
    params = {
        'securityToken': SECURITY_TOKEN,
        'documentType': 'A65',
        'processType': 'A16',
        'outBiddingZone_Domain': 'FILL_IN',  # used for Load data
        'periodStart': periodStart,  # in the format YYYYMMDDHHMM
        'periodEnd': periodEnd  # in the format YYYYMMDDHHMM
    }

    chunk_intervals = get_chunks_between_dates(periodStart, periodEnd)

    def call_api_load(_region: str, _area_code: str):
        """
        Call the ENTSO-E API to fetch load data for a specified region and time range.

        Parameters:
        - _region (str): The name of the region for which load data is being fetched.
        - _area_code (str): The area code corresponding to the specified region.
        """
        LOGGER.info(f'{_region} Fetching data...')
        params_region = params.copy()
        params_region['outBiddingZone_Domain'] = _area_code

        df_total = pd.DataFrame(columns=['StartTime', 'EndTime', 'AreaID', 'UnitName', 'Load'])

        for interval in chunk_intervals:
            # Use the requests library to get data from the API for the specified time range
            params_region["periodStart"], params_region["periodEnd"] = interval
            response_content = perform_get_request(URL_API, params_region)

            # Response content is a string of XML data
            df = xml_to_load_dataframe(response_content)

            df_total = pd.concat([df_total, df])

        df_total["Country"] = _region
        # Save the DataFrame to a CSV file
        LOGGER.info(
            f"[{_region}] Writing {df_total['StartTime'].unique().shape[0]} points in {output_path}/load_{_region}.csv...")
        df_total.to_csv(f'{PROJECT_PATH}/{output_path}/load_{_region}.csv', index=False)

    threads = {}
    # Loop through the regions and get data for each region
    for region, area_code in regions.items():
        thread = threading.Thread(target=call_api_load, args=(region, area_code))
        thread.start()
        threads[region] = thread

    for region, thread in threads.items():
        thread.join()
        LOGGER.info(f"[{region}] Load Data process has finished successfully.")

    LOGGER.info(f"{'<' * 10} Load Data got successfully for all countries. {'>' * 10}")
    return


def get_gen_data_from_entsoe(regions, periodStart='202302240000', periodEnd='202303240000', output_path='./data'):
    """
    Get generation data from the ENTSO-E API for specified regions and time range.

    Parameters:
    - regions (dict): A dictionary mapping region names to corresponding area codes.
    - periodStart (str): The start date and time for the time range in the format YYYYMMDDHHMM.
    - periodEnd (str): The end date and time for the time range in the format YYYYMMDDHHMM.
    - output_path (str): The path to the directory where the generated CSV files will be saved.
    """

    LOGGER.info("Starting the generation data's extraction...")

    # General parameters for the API
    params = {
        'securityToken': SECURITY_TOKEN,
        'documentType': 'A75',
        'processType': 'A16',
        'outBiddingZone_Domain': 'FILL_IN',  # used for Load data
        'in_Domain': 'FILL_IN',  # used for Generation data
        'periodStart': periodStart,  # in the format YYYYMMDDHHMM
        'periodEnd': periodEnd  # in the format YYYYMMDDHHMM
    }

    chunk_intervals = get_chunks_between_dates(periodStart, periodEnd)

    def call_api_gen(_region, _area_code):
        LOGGER.info(f'[{_region}] Fetching data...')
        params_region = params.copy()
        params_region['outBiddingZone_Domain'] = _area_code
        params_region['in_Domain'] = _area_code

        dfs_total = {}
        for interval in chunk_intervals:
            # Use the requests library to get data from the API for the specified time range
            params_region["periodStart"], params_region["periodEnd"] = interval
            LOGGER.info(
                f"[{_region}] Collecting data from {params_region['periodStart']} to {params_region['periodEnd']}")
            # Use the requests library to get data from the API for the specified time range
            response_content = perform_get_request(URL_API, params_region)

            # Response content is a string of XML data
            dfs = xml_to_gen_data(response_content)

            if not dfs_total:
                dfs_total = dfs.copy()
            else:
                for psr_type, df in dfs.items():
                    dfs_total[psr_type] = pd.concat([dfs_total[psr_type], df]) if psr_type in dfs_total else df.copy()

        # Save the dfs to CSV files
        for psr_type, df in dfs_total.items():
            df["Country"] = _region
            # Save the DataFrame to a CSV file
            LOGGER.info(
                f"[{_region}] - Writing {df['StartTime'].unique().shape[0]} points in "
                f"{output_path}/gen_{_region}_{psr_type}.csv...")
            df.to_csv(f'{PROJECT_PATH}/{output_path}/gen_{_region}_{psr_type}.csv', index=False)

    threads = {}
    # Loop through the regions and get data for each region
    for region, area_code in regions.items():
        thread = threading.Thread(target=call_api_gen, args=(region, area_code))
        thread.start()
        threads[region] = thread

    for region, thread in threads.items():
        thread.join()
        LOGGER.info(f"[{region}] Gen Data process has finished successfully.")

    LOGGER.info(f"{'<' * 10} Gem Data got successfully for all countries. {'>' * 10}")

    return


def parse_arguments():
    """
    Parse command-line arguments for the data ingestion script.

    Returns:
    - argparse.Namespace: An object containing parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description='Data ingestion script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--start_time',
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'),
        default=datetime.datetime(2022, 1, 1),
        help='Start time for the data to download, format: YYYY-MM-DD'
    )
    parser.add_argument(
        '--end_time',
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'),
        default=datetime.datetime(2023, 1, 1),
        help='End time for the data to download, format: YYYY-MM-DD'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./data',
        help='Name of the output file'
    )
    return parser.parse_args()


def gen_raw_data(regions, output_path, start_date, end_date):
    """
    Generate raw data by combining generation and load data for specified regions and time range.

    Parameters:
    - regions (dict): A dictionary mapping region names to corresponding keys.
    - output_path (str): The path to the directory containing the generation and load data files.
    - start_date (str): The start date for the time range.
    - end_date (str): The end date for the time range.
    """

    LOGGER.info("Starting raw_data process...")
    gen_df = pd.DataFrame()
    load_df = pd.DataFrame()
    for filename in os.listdir(output_path):

        if re.match(f"gen_({'|'.join(regions.keys())})" + r"_B\d{2}.csv", filename):
            df = pd.read_csv(f"{output_path}/{filename}", header=0, sep=",", low_memory=False)
            gen_df = pd.concat([gen_df, df])

        if re.match(f"load_({'|'.join(regions.keys())}).csv", filename):
            df = pd.read_csv(f"{output_path}/{filename}", header=0, sep=",", low_memory=False)
            load_df = pd.concat([load_df, df])

    gen_df_pivot = gen_df.pivot(index=["Country", 'StartTime', 'EndTime', 'AreaID', 'UnitName'],
                                columns='PsrType', values='quantity').reset_index()

    raw_df = gen_df_pivot.merge(load_df, on=["Country", "StartTime", "EndTime", "AreaID", "UnitName"], how="outer")

    LOGGER.info("Ensuring that all UnitName = 'MAW'...")
    assert raw_df[raw_df.UnitName != "MAW"].shape[0] == 0

    LOGGER.info("Ensuring that all data is between both dates")
    raw_df = raw_df[(raw_df['StartTime'] < end_date) & (raw_df['StartTime'] >= start_date)]

    LOGGER.info(f"Writing {raw_df.shape[0]} records {PROJECT_PATH}/{output_path}/raw_data.csv")
    raw_df.to_csv(f'{PROJECT_PATH}/{output_path}/raw_data.csv', index=False)
    LOGGER.info("Raw data successfully generated.")


def main(start_time, end_time, output_path):
    regions = {
        'HU': '10YHU-MAVIR----U',
        'IT': '10YIT-GRTN-----B',
        'PO': '10YPL-AREA-----S',
        'SP': '10YES-REE------0',
        'UK': '10Y1001A1001A92E',
        'DE': '10Y1001A1001A83F',
        'DK': '10Y1001A1001A65H',
        'SE': '10YSE-1--------K',
        'NE': '10YNL----------L',
    }
    LOGGER.info("Starting...")

    # Transform start_time and end_time to the format required by the API: YYYYMMDDHHMM
    start_time_str = start_time.strftime('%Y%m%d%H%M')
    end_time_str = end_time.strftime('%Y%m%d%H%M')

    start_total = time.monotonic()

    # Get Load data from ENTSO-E
    start = time.monotonic()
    LOGGER.info("Starting the process to get the load data...")
    get_load_data_from_entsoe(regions, start_time_str, end_time_str, output_path)
    end = time.monotonic()
    LOGGER.info(f"Duration to get the load data from entsoe: {round(end - start, 2)} seconds")

    # Get Generation data from ENTSO-E
    start = time.monotonic()
    get_gen_data_from_entsoe(regions, start_time_str, end_time_str, output_path)
    end = time.monotonic()
    LOGGER.info(f"Duration to get the gen data from entsoe: {round(end - start, 2)} seconds")

    # Generate raw_df.csv from extraction
    start = time.monotonic()
    gen_raw_data(regions, output_path, start_time.strftime('%Y-%m-%d'), end_time.strftime('%Y-%m-%d'))
    end = time.monotonic()
    LOGGER.info(
        f"Duration to build the raw_data.csv by transforming the data collected: {round(end - start, 2)} seconds")

    end_total = time.monotonic()
    LOGGER.info(f"Data Ingestion total duration: {round(end_total - start_total, 2)} seconds")


if __name__ == "__main__":
    args = parse_arguments()
    main(args.start_time, args.end_time, args.output_path)
