# littletable - a Python module to give ORM-like access to a collection of objects
[![Build Status](https://travis-ci.org/ptmcg/littletable.svg?branch=master)](https://travis-ci.org/ptmcg/littletable) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ptmcg/littletable/master)

The `littletable` module provides a low-overhead, schema-less, in-memory database access to a collection 
of user objects. `littletable` provides a `DataObject` class for ad hoc creation of semi-immutable objects 
that can be stored in a `littletable` `Table`. Tables can also contain user-defined objects, using those 
objects' `__dict__`, `__slots__`, or `_fields` mappings to access object attributes.

In addition to basic ORM-style insert/remove/query/delete access to the contents of a `Table`, `littletable` offers:
* simple indexing for improved retrieval performance, and optional enforcing key uniqueness 
* access to objects using indexed attributes 
* simplified joins using '+' operator syntax between annotated Tables 
* the result of any query or join is a new first-class littletable Table 

littletable Tables do not require an upfront schema definition, but simply work off of the attributes in 
the stored values, and those referenced in any query parameters.

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
    for item in catalog.where(lambda o: o.unitprice>10):
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
