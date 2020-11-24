#
# littletable_demo.py
#
# Copyright 2010, Paul T. McGuire
#
from __future__ import print_function

from littletable import Table
from collections import namedtuple
import sys

Customer = namedtuple("Customer", "id name")
CatalogItem = namedtuple("CatalogItem", "sku descr unitofmeas unitprice")

customers = Table("customers")
customers.create_index("id", unique=True)
customer_data = """\
id,name
0010,George Jetson
0020,Wile E. Coyote
0030,Jonny Quest"""
customers.csv_import(customer_data, row_class=Customer)

catalog = Table("catalog")
catalog.create_index("sku", unique=True)
catalog_data = """\
sku,descr,unitofmeas,unitprice
BRDSD-001,Bird seed,LB,3
BBS-001,Steel BB's,LB,5
MGNT-001,Magnet,EA,8
MAGLS-001,Magnifying glass,EA,12
ANVIL-001,1000lb anvil,EA,100
ROPE-001,1 in. heavy rope,100FT,10
ROBOT-001,Domestic robot,EA,5000"""
catalog.csv_import(catalog_data, row_class=CatalogItem, transforms={'unitprice': int})

wishitems = Table("wishitems")
wishitems.create_index("custid")
wishitems.create_index("sku")
# there is no user-defined type for these items, just use DataObjects
wishlist_data = """\
custid,sku
0030,MAGLS-001
0020,MAGLS-001
0020,ANVIL-001
0020,ROPE-001
0020,BRDSD-001
0020,BBS-001
0020,MAGNT-001
0030,MAGNT-001
0030,ROBOT-001
0010,ROBOT-001"""
wishitems.csv_import(wishlist_data)

# print a particular customer name
print(customers.by.id["0030"].name)
print()

# print all items sold by the pound
for item in catalog.where(unitofmeas="LB"):
    print(item.sku, item.descr)
print()

# if querying on an indexed item, use ".by.attribute-name[key]"
catalog.create_index("unitofmeas")
for item in catalog.by.unitofmeas["LB"]:
    print(item.sku, item.descr)
print()

# print all items that cost more than 10
for item in catalog.where(lambda ob: ob.unitprice > 10):
    print(item.sku, item.descr, item.unitprice)
print()

# join tables to create queryable wishlists collection - the following are all equivalent
wishlists = (customers.join_on("id") + wishitems.join_on("custid")).join_on("sku") + catalog.join_on("sku")
wishlists = (customers.join_on("id") + wishitems.join_on("custid")).join_on("sku") + catalog
wishlists = catalog + (customers.join_on("id") + wishitems.join_on("custid")).join_on("sku")
wishlists = catalog.join_on("sku") + (customers.join_on("id") + wishitems.join_on("custid"))
wishlists = customers.join_on("id") + wishitems.join_on("custid") + catalog.join_on("sku")
print(wishlists().table_name)
print(wishlists()("wishlists").table_name)

# print all wishlist items with price > 10 (use Tabe.gt instead of lambda)
# bigticketitems = wishlists().where(lambda ob : ob.unitprice > 10)
bigticketitems = wishlists().where(unitprice=Table.gt(10))
for bti in bigticketitems:
    print(bti)
print()

# list all wishlist items by customer, then in descending order by unit price
for item in wishlists().sort("custid, unitprice desc"):
    print(item)
print()

# display formatted tabular output (only on Python 3.6 and later)
if sys.version_info >= (3, 6):
    wishlists().sort("custid, unitprice desc")("Wishlists").select("custid name sku descr").present()

# create simple pivot table, grouping wishlist data by customer name
wishlistsdata = wishlists()
wishlistsdata.create_index("name")
pivot = wishlistsdata.pivot("name")
pivot.dump(row_fn=lambda o: "%s %s" % (o.sku, o.descr))
print()

# pivot on both sku number and customer name, giving tabular output
piv2 = wishlistsdata.pivot("sku name")
piv2.dump_counts()
print()

# pivot on both sku number and customer name, giving tabular output
# tabulate by sum(unitprice) for all items in each pivot table cell
piv2.dump_counts(count_fn=lambda recs: sum(r.unitprice for r in recs))
print()
