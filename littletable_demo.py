#
# littletable_demo.py
#
# Copyright 2010, Paul T. McGuire
#

from littletable import Table, DataObject

customers = Table("customers")
customers.create_index("id", unique=True)
customers.insert(DataObject(id="0010", name="George Jetson"))
customers.insert(DataObject(id="0020", name="Wile E. Coyote"))
customers.insert(DataObject(id="0030", name="Jonny Quest"))

# print a particular customer name
print customers.id["0030"].name
print

catalog = Table("catalog")
catalog.create_index("sku", unique=True)
catalog.insert(DataObject(sku="BRDSD-001", descr="Bird seed", unitofmeas="LB",unitprice=3))
catalog.insert(DataObject(sku="MAGNT-001", descr="Magnet", unitofmeas="EA",unitprice=8))
catalog.insert(DataObject(sku="MAGLS-001", descr="Magnifying glass", unitofmeas="EA",unitprice=12))
catalog.insert(DataObject(sku="ANVIL-001", descr="1000lb anvil", unitofmeas="EA",unitprice=100))

wishitems = Table("wishitems")
wishitems.create_index("custid")
wishitems.create_index("sku")
wishitems.insert(DataObject(custid="0030", sku="MAGLS-001"))
wishitems.insert(DataObject(custid="0020", sku="ANVIL-001"))
wishitems.insert(DataObject(custid="0020", sku="BRDSD-001"))
wishitems.insert(DataObject(custid="0020", sku="MAGNT-001"))
wishitems.insert(DataObject(custid="0030", sku="MAGNT-001"))

# print all items sold by the pound
for item in catalog.query(unitofmeas="LB"):
    print item.sku, item.descr
print

# print all items that cost more than 10
for item in catalog.where(lambda ob : ob.unitprice>10):
    print item.sku, item.descr, item.unitprice
print

# join tables to create queryable wishlists collection - the following are all equivalent
wishlists = (customers.join_on("id") + wishitems.join_on("custid")).join_on("sku") + catalog.join_on("sku")
wishlists = (customers.join_on("id") + wishitems.join_on("custid")).join_on("sku") + catalog
wishlists = catalog + (customers.join_on("id") + wishitems.join_on("custid")).join_on("sku")
wishlists = catalog.join_on("sku") + (customers.join_on("id") + wishitems.join_on("custid"))
wishlists = customers.join_on("id") + wishitems.join_on("custid") + catalog.join_on("sku")
print wishlists().table_name
print wishlists()("wishlists").table_name

# print all wishlist items with price > 10
bigticketitems = wishlists().where(lambda ob : ob.unitprice > 10)
for bti in bigticketitems:
    print bti
print

# list all wishlist items by customer, then in descending order by unit price
for item in wishlists().query(_orderby="custid, unitprice desc"):
    print item
print

# create simple pivot table, grouping wishlist data by customer name
wishlistsdata = wishlists()
wishlistsdata.create_index("name")
pivot = wishlistsdata.pivot("name")
pivot.dump(row_fn=lambda o:"%s %s" % (o.sku,o.descr))
print

# pivot on both sku number and customer name, giving tabular output
piv2 = wishlistsdata.pivot("sku name")
print piv2.dump_counts()
