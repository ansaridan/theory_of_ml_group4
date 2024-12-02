import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import logging
import datetime as dt

# Set up logger
log = logging.getLogger("etl")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()  # This will print to the console
handler.setLevel(logging.INFO)
log.addHandler(handler)

DATA_PATH = Path() / "data"
AIRLINE_CODES_FILENAME = "airline_codes_map.csv"
AIRPORT_CODES_FILENAME = "airport_id_map.csv"

SELECTED_COLS = [
    # flight identifiers / general data
    "FlightDate", "Tail_Number", 
    # "Flight_Number_Reporting_Airline",
    # "Flights", 
    "Distance", # "DistanceGroup",
    #"FirstDepTime", "TotalAddGTime", "LongestAddGTime",
    # departure, arrival time
    "CRSDepTime", "DepTime", "DepTimeBlk",
    "CRSArrTime", "ArrTime", "ArrTimeBlk",
    "ActualElapsedTime", # "AirTime", 
    # airline identifiers
    "Reporting_Airline", # "DOT_ID_Reporting_Airline", "IATA_CODE_Reporting_Airline",
    # origin
    "Origin", "OriginAirportID", "OriginCityName", # "OriginAirportSeqID", "OriginCityMarketID", "Origin", "OriginCityName", "OriginState", "OriginStateFips", "OriginStateName", "OriginWac",
    # destination
    "Dest", "DestAirportID", "DestCityName", # "DestAirportSeqID", "DestCityMarketID", "Dest", "DestCityName", "DestState", "DestStateFips", "DestStateName", "DestWac",
    # delay data
    "DepDelay","DepDelayMinutes","DepDel15","DepartureDelayGroups",
    "ArrDelay","ArrDelayMinutes","ArrDel15", "ArrivalDelayGroups",
    "CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay",
    # time spent data
    # "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn",
    # cancellation data
    "Cancelled", "CancellationCode", 
    # # diversion data
    # "Diverted", "DivAirportLandings", "DivReachedDest", "DivActualElapsedTime", "DivArrDelay", "DivDistance",
    # "Div1Airport","Div1AirportID","Div1AirportSeqID","Div1WheelsOn","Div1TotalGTime","Div1LongestGTime","Div1WheelsOff",
    # "Div1TailNum","Div2Airport","Div2AirportID","Div2AirportSeqID","Div2WheelsOn","Div2TotalGTime","Div2LongestGTime","Div2WheelsOff",
    # "Div2TailNum","Div3Airport","Div3AirportID","Div3AirportSeqID","Div3WheelsOn","Div3TotalGTime","Div3LongestGTime","Div3WheelsOff",
    # "Div3TailNum","Div4Airport","Div4AirportID","Div4AirportSeqID","Div4WheelsOn","Div4TotalGTime","Div4LongestGTime","Div4WheelsOff",
    # "Div4TailNum","Div5Airport","Div5AirportID","Div5AirportSeqID","Div5WheelsOn","Div5TotalGTime","Div5LongestGTime","Div5WheelsOff","Div5TailNum",
]

def calculate_time_difference(start_time, end_time):
    if pd.isnull(start_time) or pd.isnull(end_time):
        return np.nan
    ARBITRATY_DATE = dt.date(1900, 1, 1)
    start_dt = dt.datetime.combine(ARBITRATY_DATE, start_time)
    end_dt = dt.datetime.combine(ARBITRATY_DATE, end_time)
    if end_dt < start_dt:
        end_dt += dt.timedelta(days=1)
    return (end_dt - start_dt).total_seconds() / 60

def _ingest_data(sample_frac=1.0) -> pd.DataFrame:
    """Ingests the data from the CSV files and returns a DataFrame."""

    # Define the regex pattern for the filenames
    pattern = re.compile(r"^.*flights.*$")

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Loop through all files in the directory
    for file in DATA_PATH.glob("*.csv"):
        if pattern.match(file.name):
            log.info(f"reading {file}")
            # Read the CSV and append to the list
            df_month = pd.read_csv(file, low_memory=False)
            df_month = df_month.sample(frac=sample_frac, random_state=42)
            dataframes.append(df_month[SELECTED_COLS])
        else:
            log.info(f"skipped {file}")

    # Concatenate all DataFrames into one
    if dataframes:
        df_raw = pd.concat(dataframes, ignore_index=True)
    else:
        df_raw = pd.DataFrame()  # Empty DataFrame if no matching files found
    
    return df_raw

def _enrich_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Adds additional data from other files to the flight data and formats the data."""
    
    df = df_raw.copy()
    # # format date columns
    for date_col in ["FlightDate"]:
        df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d")
    for military_time_col in ["CRSDepTime", "DepTime", "CRSArrTime", "ArrTime"]:
        df[military_time_col] = pd.to_datetime(df[military_time_col].astype(str).str.zfill(4), format="%H%M", errors="coerce").dt.time
    df['ScheduledDurationMinutes'] = df.apply(lambda row: calculate_time_difference(row['CRSDepTime'], row['CRSArrTime']), axis=1)
    df["month"] = df["FlightDate"].dt.month
    df["is_weekend"] = df["FlightDate"].dt.dayofweek.isin([5, 6])
    df["day_of_week"] = df["FlightDate"].dt.dayofweek.astype(str) + "_" + df["FlightDate"].dt.day_name()
    df["hour_of_day"] = [time.hour for time in df["CRSDepTime"]]

    # map cancellation reasons
    cancellation_code_map = dict(
        A="Carrier Caused",
        B="Weather",
        C="National Aviation System",
        D="Security",
    )
    df.CancellationCode = df.CancellationCode.map(cancellation_code_map)

    # join in airline names by code
    airline_codes_map = pd.read_csv(DATA_PATH / AIRLINE_CODES_FILENAME)
    df = df.join(airline_codes_map.set_index("Reporting_Airline"), on="Reporting_Airline")

    # join in airport names by code
    # https://www.transtats.bts.gov/FieldInfo.asp?Svryq_Qr5p=b4vtv0%FDNv42146%FP%FDNv42146%FDVQ.%FDN0%FDvqr06vsvpn6v10%FD07zor4%FDn55vt0rq%FDoB%FDhf%FDQbg%FD61%FDvqr06vsB%FDn%FD70v37r%FDnv42146.%FD%FDh5r%FD6uv5%FDsvryq%FDs14%FDnv42146%FDn0nyB5v5%FDnp4155%FDn%FD4n0tr%FD1s%FDBrn45%FDorpn75r%FDn0%FDnv42146%FDpn0%FDpun0tr%FDv65%FDnv42146%FDp1qr%FDn0q%FDnv42146%FDp1qr5%FDpn0%FDor%FD4r75rq.&Svryq_gB2r=a7z&Y11x72_gnoyr=Y_NVecbeg_VQ&gnoyr_VQ=FMF&flf_gnoyr_anzr=g_gEDD_ZNeXRg_NYY_PNeeVRe&fB5_Svryq_anzr=beVTVa_NVecbeg_VQ
    # NOTE: regions are defined from this CSV https://github.com/cphalpert/census-regions/blob/master/us%20census%20bureau%20regions%20and%20divisions.csv
    airport_id_map = pd.read_csv(DATA_PATH / AIRPORT_CODES_FILENAME)
    df = df.join(airport_id_map.set_index("Code").rename(columns={col:f"Origin{col.replace("_", "")}" for col in airport_id_map.columns if col != "Code"}), on="OriginAirportID")
    df = df.join(airport_id_map.set_index("Code").rename(columns={col:f"Dest{col.replace("_", "")}" for col in airport_id_map.columns if col != "Code"}), on="DestAirportID")

    return df
    

def get_flight_data(sample_frac=1.0) -> pd.DataFrame:
    """Returns the enriched flight data."""
    df_raw = _ingest_data(sample_frac=sample_frac)
    df = _enrich_data(df_raw)
    return df