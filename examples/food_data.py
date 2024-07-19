#
# food_data.py
#
# Work with food data to do some Table functions.
# (Using present() requires installation of the rich module.)
#
from pathlib import Path

import littletable as lt

data_dir = Path("food_data_set")
food_data: lt.Table = lt.csv_import(
    data_dir / "combined_food_data.csv",
    transforms={"*": lt.Table.convert_numeric}
)("Food Data")

# create some useful indexes
food_data.create_index("id", unique=True)
food_data.create_search_index("food")

# find foods that have "cheese" in the name, but not "pizza" or "bread"
search_query = "cheese --pizza --bread"
food_data.search.food(search_query).orderby("id").select("id food caloric_value").present()

# 10 foods with the highest caloric value
food_data.orderby("caloric_value desc")[:10].select("id food caloric_value").present()
