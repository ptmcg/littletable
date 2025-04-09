#
# time_zone_db.py
#
# Read data from a .zip archive containing 2 CSV files, describing
# time zone definitions and country code definitions.
#
# Uses data fetched from https://timezonedb.com/files/TimeZoneDB.csv.zip,
# The database is licensed under Creative Commons Attribution 3.0 License.
# More info at https://timezonedb.com/download
#
from datetime import datetime, timedelta
from pathlib import Path
import sys

import littletable as lt


tzdb_zip_file = Path(__file__).parent / "TimeZoneDB.csv.zip"

if not tzdb_zip_file.exists():
    print("File not found: Download TimeZoneDB.csv.zip from https://timezonedb.com/files/TimeZoneDB.csv.zip")
    sys.exit()


# read in all country codes and display the first 20
country_codes = lt.csv_import(
    tzdb_zip_file,
    zippath="country.csv",
    fieldnames="country_code,country_name".split(","),
)
country_codes[:20].present()


def str_to_datetime(s: str) -> datetime:
    """
    Function to transform timestamp seconds into a datetime object.
    Need special handling since times before Jan 1, 1970 are recorded as
    negative values, which datetime.fromtimestamp cannot handle directly.
    """
    timestamp = int(s)
    if timestamp < 0:
        return datetime.fromtimestamp(0) + timedelta(seconds=timestamp)
    else:
        return datetime.fromtimestamp(timestamp)


# read in all timezone definitions and present the first 20
time_zones = lt.csv_import(
    tzdb_zip_file,
    zippath="time_zone.csv",
    fieldnames="zone_name,country_code,abbreviation,time_start,gmt_offset,dst".split(","),
    transforms={
        "dst": lambda x: x == "1",
        "gmt_offset": int,
        "time_start": str_to_datetime
    },
)
time_zones[:20].present(
    width=120,
    caption=f"Total {len(time_zones):,} time zone records",
    caption_justify="left"
)

# query for time zone records for America/Los_Angeles in 2025
start_time = datetime(2025, 1, 1)
end_time = datetime(2025, 12, 31)
time_zones.where(
    zone_name="America/Los_Angeles",
    time_start=lt.Table.in_range(start_time, end_time)
)("Time zone records for America/Los_Angeles in 2025").present(width=120)

# how many different time zone names are there?
time_zone_names = list(time_zones.all.zone_name.unique)
print("Total time zone names:", len(time_zone_names))
