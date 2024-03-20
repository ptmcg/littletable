#
# peps.py
#
# Using PEP data extracted from Python.org, this program demonstrates
# pulling data from a compressed JSON file, and accessing PEP entries
# by id, year created, and text search of the PEP abstracts.
#
# Copyright Paul McGuire, 2022
#
import json
from pathlib import Path
import littletable as lt


# import PEP data from JSON, converting id's to ints and created
# date stings to Python datetimes
peps = lt.Table().json_import(
    Path(__file__).parent / "peps.json.zip",
    transforms={
        "id": int,
        "created": lt.Table.parse_date("%d-%b-%Y"),
    }
)

# print metadata of imported records
print(peps.info())

# access records by unique PEP id
peps.create_index("id", unique=True)
print(peps.by.id[20].title)

# add a numeric "year" field, and index it
peps.add_field("year", lambda pep: pep.created.year)
peps.create_index("year")

# present PEPs created in 2016
peps.by.year[2016]("PEPs Created in 2016").select("id python_version title").present()

# how many PEPs since 2020?
print("Number of PEPs since 2020", len(peps.by.year[2020:]))
print()

# pivot by year and dump counts, or present as nice table
peps.pivot("year").dump_counts()
peps.pivot("year").as_table().present()

# create full text search on PEP abstracts
peps.create_search_index("abstract")

# search for PEPs referring to the walrus operator
walrus_pep = peps.search.abstract("walrus", as_table=True)("'walrus' Search Results")
walrus_pep.select("id title year").present()
print(walrus_pep.select("id title year").json_export())

# search for PEPs referring to GvR or Guido or BDFL
bdfl_peps = peps.search.abstract("gvr guido bdfl", as_table=True)("GvR PEPs")
bdfl_peps.orderby("id")
bdfl_peps.select("id title year").present()

# define a custom JSON encoder for datetime.date field
class JsonDateEncoder(json.JSONEncoder):
    def default(self, o):
        import datetime
        if isinstance(o, datetime.date):
            return str(o)
        return super().default(o)


print(bdfl_peps.select("id title created").json_export(json_encoder=(JsonDateEncoder,)))
