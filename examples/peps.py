import littletable as lt

# import PEP data from JSON, converting id's to ints and created
# date stings to Python datetimes
peps = lt.Table().json_import(
    "peps.json.zip",
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

# add a numeric "year" field
peps.add_field("year", lambda pep: pep.created.year)
peps.create_index("year")

# present PEPs created in 2016
peps.by.year[2016].select("id title").present()

# pivot by year and dump counts, or present as nice table
peps.pivot("year").dump_counts()
peps.pivot("year").as_table().present()

# create full text search on PEP abstracts
peps.create_search_index("abstract")

walrus_pep = peps.search.abstract("walrus")[0][0]
print(f"{walrus_pep.id} {walrus_pep.title} {walrus_pep.year} ")

# search for PEPs referring to GvR or Guido or BDFL
# (as_table requires littletable 2.1.1)
if lt.__version_info__[:3] >= (2, 1, 1):
    bdfl_peps = peps.search.abstract("gvr guido bdfl", as_table=True)
    bdfl_peps.sort("id").select("id title year").present()
