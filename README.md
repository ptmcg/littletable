# littletable - a Python module to give ORM-like access to a collection of objects
[![Build Status](https://travis-ci.org/ptmcg/littletable.svg?branch=master)](https://travis-ci.org/ptmcg/littletable) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ptmcg/littletable/master)

- [Introduction](#introduction)
- [Importing data from CSV files](#importing-data-from-csv-files)
- [Tabular output](#tabular-output)
- [For More Info](#for-more-info)
- [Sample Demo](#sample-demo)

Introduction
------------
The `littletable` module provides a low-overhead, schema-less, in-memory database access to a collection 
of user objects. `littletable` Tables will accept Python `dict`s or any user-defined object type, including `namedtuples`, `dataclasses`,
and `types.SimpleNamespaces`, using those objects' `__dict__`, `__slots__`, or `_fields` mappings to access object
attributes and determine the Table's "columns". `littletable` also provides a `DataObject` class for easy creation of
namespace objects from Python `dict` instances. If populated with Python `dict`s, they get stored
as `SimpleNamespace`s or `DictObject`s.

In addition to basic ORM-style insert/remove/query/delete access to the contents of a `Table`, `littletable` offers:
* simple indexing for improved retrieval performance, and optional enforcing key uniqueness 
* access to objects using indexed attributes
* direct CSV import/export
* clean tabular output for data presentation
* simplified joins using `"+"` operator syntax between annotated `Table`s 
* the result of any query or join is a new first-class `littletable` `Table` 
* simple full-text search against multi-word text attributes
* access like a standard Python list to the records in a Table, including indexing/slicing, `iter`, `zip`, `len`, `groupby`, etc.
* access like a standard Python `dict` to attributes with a unique index, or like a standard Python `defaultdict(list)` to attributes with a non-unique index

`littletable` `Table`s do not require an upfront schema definition, but simply work off of the attributes in 
the stored values, and those referenced in any query parameters.


Importing data from CSV files
-----------------------------
You can easily import a CSV file into a Table using Table.csv_import():

    t = Table().csv_import("my_data.csv")

In place of a local file name, you can also specify  an HTTP url:

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    iris_table = Table('iris').csv_import(url)

You can also directly import CSV data as a string:

    catalog = Table("catalog")

    catalog_data = """\
    sku,description,unitofmeas,unitprice
    BRDSD-001,Bird seed,LB,3
    BBS-001,Steel BB's,LB,5
    MGNT-001,Magnet,EA,8"""

    catalog.csv_import(catalog_data, transforms={'unitprice': int})

Data can also be directly imported from compressed .zip, .gz, and .xz files.

Files containing JSON-formatted records can be similarly imported using `Table.json_import()`.


Tabular output
--------------
To produce a nice tabular output for a table, you can use the embedded support for
the [rich](https://github.com/willmcgugan/rich) module, `as_html()` in [Jupyter Notebook](https://jupyter.org/),
or the [tabulate](https://github.com/astanin/python-tabulate) module:

Using `table.present()` (implemented using `rich`; `present()` accepts `rich` `Table` keyword args):

    table(title_str).present(fields=["col1", "col2", "col3"])
      or
    table.select("col1 col2 col3")(title_str).present(caption="caption text", 
                                                      caption_justify="right")

Using `Jupyter Notebook`:

    from IPython.display import HTML, display
    display(HTML(table.as_html()))

Using `tabulate`:

    from tabulate import tabulate
    print(tabulate(map(vars, table), headers="keys"))

For More Info
-------------
Extended "getting started" notes at [how_to_use_littletable.md](https://github.com/ptmcg/littletable/blob/master/how_to_use_littletable.md).

Sample Demo
-----------
Here is a simple littletable data storage/retrieval example:

    from littletable import Table, DataObject

    customers = Table('customers')
    customers.create_index("id", unique=True)
    customers.insert(DataObject(id="0010", name="George Jetson"))
    customers.insert(DataObject(id="0020", name="Wile E. Coyote"))
    customers.insert(DataObject(id="0030", name="Jonny Quest"))

    catalog = Table('catalog')
    catalog.create_index("sku", unique=True)
    catalog.insert(DataObject(sku="ANVIL-001", descr="1000lb anvil", unitofmeas="EA",unitprice=100))
    catalog.insert(DataObject(sku="BRDSD-001", descr="Bird seed", unitofmeas="LB",unitprice=3))
    catalog.insert(DataObject(sku="MAGNT-001", descr="Magnet", unitofmeas="EA",unitprice=8))
    catalog.insert(DataObject(sku="MAGLS-001", descr="Magnifying glass", unitofmeas="EA",unitprice=12))

    wishitems = Table('wishitems')
    wishitems.create_index("custid")
    wishitems.create_index("sku")
    # easy to import CSV data from a string or file
    wishitems.csv_import("""\
    custid,sku
    0020,ANVIL-001
    0020,BRDSD-001
    0020,MAGNT-001
    0030,MAGNT-001
    0030,MAGLS-001
    """)

    # print a particular customer name 
    # (unique indexes will return a single item; non-unique
    # indexes will return a list of all matching items)
    print(customers.by.id["0030"].name)

    # see all customer names
    for name in customers.all.name:
        print(name)

    # print all items sold by the pound
    for item in catalog.where(unitofmeas="LB"):
        print(item.sku, item.descr)

    # print all items that cost more than 10
    for item in catalog.where(lambda o: o.unitprice > 10):
        print(item.sku, item.descr, item.unitprice)

    # join tables to create queryable wishlists collection
    wishlists = customers.join_on("id") + wishitems.join_on("custid") + catalog.join_on("sku")

    # print all wishlist items with price > 10 (can use Table.gt comparator instead of lambda)
    bigticketitems = wishlists().where(unitprice=Table.gt(10))
    for item in bigticketitems:
        print(item)

    # list all wishlist items in descending order by price
    for item in wishlists().sort("unitprice desc"):
        print(item)

    # print output as a nicely-formatted table
    wishlists().sort("unitprice desc")("Wishlists").present()

    # print output as an HTML table
    print(wishlists().sort("unitprice desc")("Wishlists").as_html())

    # print output as a Markdown table
    print(wishlists().sort("unitprice desc")("Wishlists").as_markdown())
