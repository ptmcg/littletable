#
# littletable_demo.py
#
# Copyright 2010, Paul T. McGuire
#

from littletable import Table, DataObject
from collections import namedtuple
Customer = namedtuple("Customer", "id name")
CatalogItem = namedtuple("CatalogItem", "sku descr unitofmeas unitprice")

customers = Table("customers")
customers.create_index("id", unique=True)
customers.insert(Customer(id="0010", name="George Jetson"))
customers.insert(Customer(id="0020", name="Wile E. Coyote"))
customers.insert(Customer(id="0030", name="Jonny Quest"))

catalog = Table("catalog")
catalog.create_index("sku", unique=True)
catalog.insert(CatalogItem(sku="BRDSD-001", descr="Bird seed", unitofmeas="LB",unitprice=3))
catalog.insert(CatalogItem(sku="BBS-001", descr="Steel BB's", unitofmeas="LB",unitprice=5))
catalog.insert(CatalogItem(sku="MAGNT-001", descr="Magnet", unitofmeas="EA",unitprice=8))
catalog.insert(CatalogItem(sku="MAGLS-001", descr="Magnifying glass", unitofmeas="EA",unitprice=12))
catalog.insert(CatalogItem(sku="ANVIL-001", descr="1000lb anvil", unitofmeas="EA",unitprice=100))
catalog.insert(CatalogItem(sku="ROPE-001", descr="1 in. heavy rope", unitofmeas="100FT",unitprice=10))
catalog.insert(CatalogItem(sku="ROBOT-001", descr="Domestic robot", unitofmeas="EA",unitprice=5000))

wishitems = Table("wishitems")
wishitems.create_index("custid")
wishitems.create_index("sku")
# there is no user-defined type for these items, just use DataObjects
wishitems.insert(DataObject(custid="0030", sku="MAGLS-001"))
wishitems.insert(DataObject(custid="0020", sku="MAGLS-001"))
wishitems.insert(DataObject(custid="0020", sku="ANVIL-001"))
wishitems.insert(DataObject(custid="0020", sku="ROPE-001"))
wishitems.insert(DataObject(custid="0020", sku="BRDSD-001"))
wishitems.insert(DataObject(custid="0020", sku="BBS-001"))
wishitems.insert(DataObject(custid="0020", sku="MAGNT-001"))
wishitems.insert(DataObject(custid="0030", sku="MAGNT-001"))
wishitems.insert(DataObject(custid="0030", sku="ROBOT-001"))
wishitems.insert(DataObject(custid="0010", sku="ROBOT-001"))

# print a particular customer name
print(customers.by.id["0030"].name)
print('')

# print all items sold by the pound
for item in catalog.where(unitofmeas="LB"):
    print(item.sku, item.descr)
print('')

# if querying on an indexed item, use ".by.attribute-name[key]"
catalog.create_index("unitofmeas")
for item in catalog.by.unitofmeas["LB"]:
    print(item.sku, item.descr)
print('')

# print all items that cost more than 10
for item in catalog.where(lambda ob : ob.unitprice > 10):
    print(item.sku, item.descr, item.unitprice)
print('')

# join tables to create queryable wishlists collection - the following are all equivalent
wishlists = (customers.join_on("id") + wishitems.join_on("custid")).join_on("sku") + catalog.join_on("sku")
wishlists = (customers.join_on("id") + wishitems.join_on("custid")).join_on("sku") + catalog
wishlists = catalog + (customers.join_on("id") + wishitems.join_on("custid")).join_on("sku")
wishlists = catalog.join_on("sku") + (customers.join_on("id") + wishitems.join_on("custid"))
wishlists = customers.join_on("id") + wishitems.join_on("custid") + catalog.join_on("sku")
print(wishlists().table_name)
print(wishlists()("wishlists").table_name)

# print all wishlist items with price > 10
bigticketitems = wishlists().where(lambda ob : ob.unitprice > 10)
for bti in bigticketitems:
    print(bti)
print('')

# list all wishlist items by customer, then in descending order by unit price
for item in wishlists().sort("custid, unitprice desc"):
    print(item)
print('')

# create simple pivot table, grouping wishlist data by customer name
wishlistsdata = wishlists()
wishlistsdata.create_index("name")
pivot = wishlistsdata.pivot("name")
pivot.dump(row_fn=lambda o:"%s %s" % (o.sku,o.descr))
print('')

# pivot on both sku number and customer name, giving tabular output
piv2 = wishlistsdata.pivot("sku name")
piv2.dump_counts()
print('')

# pivot on both sku number and customer name, giving tabular output
# tabulate by sum(unitprice) for all items in each pivot table cell
piv2.dump_counts(count_fn=lambda recs:sum(r.unitprice for r in recs))
print('')
