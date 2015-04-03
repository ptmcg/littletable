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

catalog = Table("catalog")
catalog.create_index("sku", unique=True)
catalog.insert(DataObject(sku="BRDSD-001", descr="Bird seed", unitofmeas="LB",unitprice=3))
catalog.insert(DataObject(sku="MAGNT-001", descr="Magnet", unitofmeas="EA",unitprice=8))
catalog.insert(DataObject(sku="MAGLS-001", descr="Magnifying glass", unitofmeas="EA",unitprice=12))
catalog.insert(DataObject(sku="ANVIL-001", descr="1000lb anvil", unitofmeas="EA",unitprice=100))
catalog.insert(DataObject(sku="ROPE-001", descr="1 in. heavy rope", unitofmeas="100FT",unitprice=10))
catalog.insert(DataObject(sku="ROBOT-001", descr="Domestic robot", unitofmeas="EA",unitprice=5000))

wishitems = Table("wishitems")
wishitems.create_index("custid")
wishitems.create_index("sku")
wishitems.insert(DataObject(custid="0030", sku="MAGLS-001"))
wishitems.insert(DataObject(custid="0020", sku="MAGLS-001"))
wishitems.insert(DataObject(custid="0020", sku="ANVIL-001"))
wishitems.insert(DataObject(custid="0020", sku="ROPE-001"))
wishitems.insert(DataObject(custid="0020", sku="BRDSD-001"))
wishitems.insert(DataObject(custid="0020", sku="MAGNT-001"))
wishitems.insert(DataObject(custid="0030", sku="MAGNT-001"))
wishitems.insert(DataObject(custid="0030", sku="ROBOT-001"))
wishitems.insert(DataObject(custid="0010", sku="ROBOT-001"))

# print a particular customer name
print customers.by.id["0030"].name
print

# print all items sold by the pound
for item in catalog.where(unitofmeas="LB"):
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
for item in wishlists().sort("custid, unitprice desc"):
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
piv2.dump_counts()
print

# pivot on both sku number and customer name, giving tabular output
# tabulate by sum(unitprice) for all items in each pivot table cell
piv2.dump_counts(count_fn=lambda recs:sum(r.unitprice for r in recs))
print
