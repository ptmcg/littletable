#
# csv_import_examples.py
#
# Some examples of importing data from CSV sources.
#
# CSV data may be imported from:
# - a Python string
# - a local .csv file
# - a .zip file containing a .csv file
# - a .gz file containing a .csv file
# - a .xz or .lzma file containing a .csv file
# - an http or https URL to a page containing raw CSV data
#
# littletable tables can also import data from:
# - TSV files (tab-separated values)
# - JSON files
#   . a single JSON list of objects, or
#   . multiple JSON objects
# - Excel spreadsheets
#

import littletable as lt

# read from CSV data in a string
catalog_data = """\
sku,descr,unitofmeas,unitprice
BRDSD-001,Bird seed,LB,3
BBS-001,Steel BB's,LB,5
MGNT-001,Magnet,EA,8
MAGLS-001,Magnifying glass,EA,12
ANVIL-001,1000lb anvil,EA,100
ROPE-001,1 in. heavy rope,100FT,10
ROBOT-001,Domestic robot,EA,5000"""
catalog = lt.csv_import(catalog_data, transforms={"unitprice": int})

# read from a CSV file
data = lt.csv_import("my_data.csv")

# read from a ZIP file containing a single CSV file
data = lt.csv_import("my_data.csv.zip")

# read from a GZIP file containing a single CSV file
data = lt.csv_import("my_data.csv.gz")

# read from CSV data in a remote URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
iris_transforms = dict.fromkeys(
    ["petal-length", "petal-width", "sepal-length", "sepal-width"], float
)
iris_table = lt.csv_import(
    url, fieldnames=names, transforms=iris_transforms
)("iris")
print(iris_table.info())

# accumulate data from multiple CSVs into a single table, and then resave as a single CSV
from pathlib import Path
csv_files = Path(".").glob("*.csv")

all_data = lt.Table()
for csv in csv_files:
    # repeated calls to csv_import appends data to an existing table
    all_data.csv_import(csv)

all_data.csv_export("accumulated_data.csv")
