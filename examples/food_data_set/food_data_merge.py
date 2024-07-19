#
# food_data_merge.py
#
# Work with data files describing nutritional content of common foods,
# downloaded from https://www.kaggle.com/datasets/utsavdey1410/food-nutrition-dataset.
#
# Rename columns to be valid Python identifiers, and re-rank so that
# id's are unique.
#

from pathlib import Path

import littletable as lt

data_dir = Path(__file__).parent

# read all CSVs into a single table - multiple calls to csv_import
# on the same Table will concatenate all into a single group
foods = lt.Table()
for data_file in data_dir.glob("FOOD-DATA-GROUP*.csv"):
    foods.csv_import(data_file, transforms={"*": lt.Table.convert_numeric})

# re-number food items into an "id" field
foods.rank("id")

# convert field names with capital letters and spaces to snake case,
# to make them easier to work with as valid Python identifiers
field_names = foods.info()["fields"]
py_names = [nm.lower().replace(" ", "_") for nm in field_names]
for py_name, field_name in zip(py_names, field_names):
    if not py_name:
        continue
    foods.compute_field(py_name, field_name)

# change the initial field name from "" to "id", and remove "unnamed" field
py_names[0] = "id"
py_names.remove("unnamed:_0")

# build single CSV file from all data
foods.select(py_names).csv_export(data_dir / "combined_food_data.csv")
