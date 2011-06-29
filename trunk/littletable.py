#
# littletable.py
# 
# littletable is a simple in-memory database for ad-hoc or user-defined objects,
# supporting simple query and join operations - useful for ORM-like access
# to a collection of data objects, without dealing with SQL
#
#
# Copyright (c) 2010  Paul T. McGuire
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

__doc__ = """\

C{littletable} - a Python module to give ORM-like access to a collection of objects

The C{littletable} module provides a low-overhead, schema-less, in-memory database access to a 
collection of user objects.  C{littletable} provides a L{DataObject} class for ad hoc creation
of semi-immutable objects that can be stored in a C{littletable} L{Table}.

In addition to basic ORM-style insert/remove/query/delete access to the contents of a 
Table, C{littletable} offers:
 - simple indexing for improved retrieval performance, and optional enforcing key uniqueness
 - access to objects using indexed attributes
 - simplified joins using '+' operator syntax between annotated Tables
 - the result of any query or join is a new first-class C{littletable} Table

C{littletable} Tables do not require an upfront schema definition, but simply work off of the
attributes in the stored values, and those referenced in any query parameters.

Here is a simple C{littletable} data storage/retrieval example::

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
    wishitems.insert(DataObject(custid="0020", sku="ANVIL-001"))
    wishitems.insert(DataObject(custid="0020", sku="BRDSD-001"))
    wishitems.insert(DataObject(custid="0020", sku="MAGNT-001"))
    wishitems.insert(DataObject(custid="0030", sku="MAGNT-001"))
    wishitems.insert(DataObject(custid="0030", sku="MAGLS-001"))

    # print a particular customer name 
    # (unique indexes will return a single item; non-unique
    # indexes will return a list of all matching items)
    print customers.id["0030"].name

    # print all items sold by the pound
    for item in catalog.query(unitofmeas="LB"):
        print item.sku, item.descr

    # print all items that cost more than 10
    for item in catalog.where(lambda o : o.unitprice>10):
        print item.sku, item.descr, item.unitprice

    # join tables to create queryable wishlists collection
    wishlists = customers.join_on("id") + wishitems.join_on("custid") + catalog.join_on("sku")

    # print all wishlist items with price > 10
    bigticketitems = wishlists().where(lambda ob : ob.unitprice > 10)
    for item in bigticketitems:
        print item

    # list all wishlist items in descending order by price
    for item in wishlists().query(_orderbydesc="unitprice"):
        print item
"""

__version__ = "0.4"
__versionTime__ = "29 Jun 2011 16:36"
__author__ = "Paul McGuire <ptmcg@users.sourceforge.net>"

import sys
from collections import defaultdict
from itertools import groupby,ifilter,islice,starmap,repeat
import csv

try:
    from itertools import product
except ImportError:
    def product(aseq,bseq):
        for a in aseq:
            for b in bseq:
                yield a,b

try:
    t = basestring
except NameError:
    basestring = str

__all__ = ["DataObject", "Table", "JoinTerm", "PivotTable"]

def _object_attrnames(obj):
    if hasattr(obj, "__dict__"):
        # normal object
        return obj.__dict__.keys()
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # namedtuple
        return obj._fields
    elif hasattr(obj, "__slots__"):
        return obj.__slots__
    else:
        raise ValueError("object with unknown attributes")

class DataObject(object):
    """A generic semi-mutable object for storing data values in a table. Attributes
       can be set by passing in named arguments in the constructor, or by setting them
       as C{object.attribute = value}. New attributes can be added any time, but updates
       are ignored.  Table joins are returned as a Table of DataObjects."""
    def __init__(self, **kwargs):
        if kwargs:
            self.__dict__.update(kwargs)
    def __repr__(self):
        return repr(self.__dict__)
    def __setattr__(self, attr, val):
        # make all attributes write-once
        if attr not in self.__dict__:
            super(DataObject,self).__setattr__(attr,val)
    def __getitem__(self, k):
        if hasattr(self,k):
            return getattr(self,k)
        else:
            raise KeyError("object has no such attribute " + k)

class _ObjIndex(object):
    def __init__(self, attr):
        self.attr = attr
        self.obs = defaultdict(list)
        self.is_unique = False
    def __setitem__(self, k, v):
        self.obs[k].append(v)
    def __getitem__(self, k):
        return self.obs.get(k,[])
    def __len__(self):
        return len(self.obs)
    def __iter__(self):
        return iter(self.obs)
    def keys(self):
        return sorted(self.obs.keys())
    def items(self):
        return self.obs.items()
    def remove(self, obj):
        try:
            k = getattr(obj, self.attr)
            self.obs[k].remove(obj)
        except (ValueError,AttributeError,KeyError):
            pass
    def __contains__(self, key):
        return key in self.obs
    def copy_template(self):
        return self.__class__(self.attr)
        
class _UniqueObjIndex(_ObjIndex):
    def __init__(self, attr, accept_none=False):
        self.attr = attr
        self.obs = {}
        self.is_unique = True
        self.accept_none = accept_none
        self.none_values = set()
    def __setitem__(self, k, v):
        if k:
            if k not in self.obs:
                self.obs[k] = v
            else:
                raise KeyError("duplicate key value %s" % k)
        else:
            self.none_values.add(v)
    def __getitem__(self, k):
        if k:
            return [self.obs.get(k)] if k in self.obs else []
        else:
            return list(self.none_values)
    def __contains__(self, k):
        if k:
            return k in self.obs
        else:
            return self.accept_none and self.none_values
    def keys(self):
        return sorted(self.obs.keys()) + ([None,] if self.none_values else [])
    def items(self):
        return [(k,[v]) for k,v in self.obs.items()]
    def remove(self, obj):
        k = getattr(obj, self.attr)
        if k:
            if k in self.obs:
                del self.obs[k]
        else:
            self.none_values.discard(obj)

class _ObjIndexWrapper(object):
    def __init__(self, ind):
        self._index = ind
    def __getattr__(self, attr):
        return getattr(self._index, attr)
    def __getitem__(self, k):
        ret = Table()
        if k in self._index:
            ret.insert_many(self._index[k])
        return ret
    def __contains__(self, k):
        return k in self._index

class _UniqueObjIndexWrapper(object):
    def __init__(self, ind):
        self._index = ind
    def __getattr__(self, attr):
        return getattr(self._index, attr)
    def __contains__(self, k):
        return k in self._index
    def __getitem__(self, k):
        if k:
            return self._index[k][0]
        else:
            ret = Table()
            if k in self._index:
                ret.insert_many(self._index[k])
            return ret
            

class Table(object):
    """Table is the main class in C{littletable}, for representing a collection of DataObjects or
       user-defined objects with publicly accessible attributes or properties.  Tables can be:
        - created, with an optional name, using standard Python L{C{Table() constructor}<__init__>}
        - indexed, with multiple indexes, with unique or non-unique values, see L{create_index}
        - queried, specifying values to exact match in the desired records, see L{query}
        - filtered (using L{where}), using a simple predicate function to match desired records;
          useful for selecting using inequalities or compound conditions
        - accessed directly for keyed values, using C{table.indexattribute[key]} - see L{__getattr__}
        - joined, using L{join_on} to identify attribute to be used for joining with another table, and
          L{join} or operator '+' to perform the actual join
        - pivoted, using L{pivot} to create a nested structure of sub-tables grouping objects
          by attribute values
        - L{imported<csv_import>}/L{exported<csv_export>} to CSV-format files
       Queries and joins return their results as new Table objects, so that queries and joins can
       be easily performed as a succession of operations.
    """
    def __init__(self, table_name=''):
        """Create a new, empty Table.
           @param table_name: name for Table
           @type table_name: string (optional)
        """
        self.table_name = table_name
        self.obs = []
        self._indexes = {}

    def __len__(self):
        """Return the number of objects in the Table."""
        return len(self.obs)
        
    def __iter__(self):
        """Create an iterator over the objects in the Table."""
        return iter(self.obs)
        
    def __getitem__(self, i):
        """Provides direct indexed/sliced access to the Table's underlying list of objects."""
        return self.obs[i]
    
    def __getattr__(self, attr):
        """A quick way to query for matching records using their indexed attributes. The attribute
           name is used to locate the index, and returns a wrapper on the index.  This wrapper provides
           dict-like access to the underlying records in the table, as in::
           
              employees.socsecnum["000-00-0000"]
              customers.zipcode["12345"]
              
           The behavior differs slightly for unique and non-unique indexes:
             - if the index is unique, then retrieving a matching object, will return just the object;
               if there is no matching object, C{KeyError} is raised
             - if the index is non-unique, then all matching objects will be returned in a new Table,
               just as if a regular query had been performed; if no objects match the key value, an empty
               Table is returned and no exception is raised.
               
           If there is no index defined for the given attribute, then C{AttributeError} is raised.
        """
        if attr in self._indexes:
            ret = self._indexes[attr]
            if isinstance(ret, _UniqueObjIndex):
                ret = _UniqueObjIndexWrapper(ret)
            if isinstance(ret, _ObjIndex):
                ret = _ObjIndexWrapper(ret)
            return ret
        raise AttributeError("Table '%s' has no index '%s'" % 
                                                (self.table_name, attr))

    def __bool__(self):
        return bool(self.obs)
    
    __nonzero__ = __bool__
    
    def __call__(self, table_name):
        """A simple way to assign a name to a table, such as those
           dynamically created by joins and queries.
           @param table_name: name for Table
           @type table_name: string
        """
        self.table_name = table_name
        return self

    def copy_template(self):
        """Create empty copy of the current table, with copies of all
           index definitions.
        """
        ret = Table(self.table_name)
        for k,v in self._indexes.items():
            ret._indexes[k] = v.copy_template()
        return ret

    def clone(self):
        """Create full copy of the current table, including table contents
           and index definitions.
        """
        ret = self.copy_template()
        ret.insert_many(self.obs)
        return ret

    def create_index(self, attr, unique=False, accept_none=False):
        """Create a new index on a given attribute.
           If C{unique} is True and records are found in the table with duplicate
           attribute values, the index is deleted and C{KeyError} is raised.

           If the table already has an index on the given attribute, then no 
           action is taken and no exception is raised.
           @param attr: the attribute to be used for indexed access and joins
           @type attr: string
           @param unique: flag indicating whether the indexed field values are 
               expected to be unique across table entries
           @type unique: boolean
           @param accept_none: flag indicating whether None is an acceptable
               value for this attribute
           @type accept_none: boolean
        """
        if attr in self._indexes:
            return
            
        if unique:
            self._indexes[attr] = _UniqueObjIndex(attr,accept_none)
        else:
            self._indexes[attr] = _ObjIndex(attr)
            accept_none = True
        ind = self._indexes[attr]
        try:
            for obj in self.obs:
                if hasattr(obj, attr):
                    obval = getattr(obj, attr) or None
                else:
                    obval = None
                if obval or accept_none:
                    ind[obval] = obj
                else:
                    raise KeyError("None is not an allowed key")
                    
        except KeyError:
            del self._indexes[attr]
            raise
    
    def delete_index(self, attr):
        """Deletes an index from the Table.  Can be used to drop and rebuild an index,
           or to convert a non-unique index to a unique index, or vice versa.
           @param attr: name of an indexed attribute
           @type attr: string
        """
        if attr in self._indexes:
            del self._indexes[attr]
            
    def insert(self, obj):
        """Insert a new object into this Table.
           @param obj: any Python object
           Objects can be constructed using the defined DataObject type, or they can
           be any Python object that does not use the Python C{__slots__} feature; C{littletable}
           introspect's the object's C{__dict__} or C{_fields} attributes to obtain join and 
           index attributes and values.
           
           If the table contains a unique index, and the record to be inserted would add
           a duplicate value for the indexed attribute, then C{KeyError} is raised, and the
           object is not inserted.
           
           If the table has no unique indexes, then it is possible to insert duplicate
           objects into the table.
           """
           
        # verify new object doesn't duplicate any existing unique index values
        uniqueIndexes = [ind for ind in self._indexes.values() if ind.is_unique]
        if any((getattr(obj, ind.attr, None) is None and not ind.accept_none) 
                or (
                hasattr(obj, ind.attr) and getattr(obj, ind.attr) in ind
                ) 
                for ind in uniqueIndexes):
            # had a problem, find which one
            for ind in uniqueIndexes:
                if (getattr(obj, ind.attr, None) is None and not ind.accept_none):
                    raise KeyError("unique key cannot be None or blank for index %s" % ind.attr, obj)
                if getattr(obj, ind.attr) in ind:
                    raise KeyError("duplicate unique key value '%s' for index %s" % (getattr(obj,ind.attr), ind.attr), obj)

        self.obs.append(obj)
        for attr, ind in self._indexes.items():
            obval = getattr(obj, attr)
            ind[obval] = obj
            
    def insert_many(self, it):
        """Inserts a collection of objects into the table."""
        for ob in it:
            self.insert(ob)

    def remove(self, ob):
        """Removes an object from the table. If object is not in the table, then
           no action is taken and no exception is raised."""
        # remove from indexes
        for attr,ind in self._indexes.items():
            ind.remove(ob)

        # remove from main object list
        self.obs.remove(ob)

    def remove_many(self, it):
        """Removes a collection of objects from the table."""
        for ob in it:
            self.remove(ob)

    def _query_attr_sort_fn(self, attr_val):
        attr,v = attr_val
        if attr in self._indexes:
            idx = self._indexes[attr]
            if v in idx:
                return len(idx[v])
            else:
                return 0
        else:
            return 1e9
        
    def query(self, **kwargs):
        """Retrieves matching objects from the table, based on given
           named parameters.  If multiple named parameters are given, then
           only objects that satisfy all of the query criteria will be returned.
           
           Special kwargs:
            - C{_orderby="attr,..."} - resulting table should sort content objects
                by the C{attr}s given in a comma-separated string; to sort in 
                descending order, reference the attribute as C{attr desc}.

           @param **kwargs: attributes for selecting records, given as additional 
              named arguments of the form C{attrname="attrvalue"}.
           @return: a new Table containing the matching objects
        """
        # extract meta keys
        flags = [(k,v) for k,v in kwargs.items() if k.startswith("_")]
        for f,v in flags:
            del kwargs[f]

        if kwargs:
            ret = self.copy_template()
            first = True
            
            # order query criteria in ascending order of number of matching items
            # for each individual given attribute; this will minimize the number 
            # of filtering records that each subsequent attribute will have to
            # handle
            kwargs = kwargs.items()
            if len(kwargs) > 1 and len(self.obs) > 100:
                kwargs = sorted(kwargs, key=self._query_attr_sort_fn)
            for k,v in kwargs:
                if k in flags:
                    continue
                if first:
                    if k in self._indexes:
                        ret.insert_many(self._indexes[k][v])
                    else:
                        ret.insert_many( r for r in self.obs 
                                        if hasattr(r,k) and getattr(r,k) == v )
                else:
                    if k in ret._indexes:
                        newret = ret.copy_template()
                        newret.insert_many(ret._indexes[k][v])
                        ret = newret
                    else:
                        retobs = ret.obs[:]
                        ret.remove_many( o for o in retobs 
                                        if not hasattr(r,k) 
                                            or (getattr(r,k) != v) )
                first = False
        else:
            ret = self.clone()
        
        for f,v in flags:
            if f == "_orderby":
                attrs = [s.strip() for s in v.split(',')]
                attr_orders = [(a.split()+['asc',])[:2] for a in attrs][::-1]
                for attr,order in attr_orders:
                     ret.obs.sort(key=lambda ob:getattr(ob,attr), reverse=(order=="desc"))
            
        return ret

    def delete(self, **kwargs):
        """Deletes matching objects from the table, based on given
           named parameters.  If multiple named parameters are given, then
           only objects that satisfy all of the query criteria will be removed.
           @param **kwargs: attributes for selecting records, given as additional 
              named arguments of the form C{attrname="attrvalue"}.
           @return: the number of objects removed from the table
        """
        if not kwargs:
            return 0
        
        affected = self.query(**kwargs)
        self.remove_many(affected)
        return len(affected)
    
    def where(self, wherefn, maxrecs=0):
        """An alternative to L{query}, using a matching predicate function to
           determine whether a given object matches the query or not.  You must use
           C{where} in place of C{query} if you want to query using inequalities or more
           complex matching criteria than simple C{attribute=value}.
           @param wherefn: a method or lambda that returns a boolean result, as in::
               
               lambda ob : ob.unitprice > 10
               
           @type wherefn: callable(object) returning boolean
           @param maxrecs: if only the first 'n' records are needed, then C{where} will 
               stop after locating 'n' matching records
           @type maxrecs: int
           @returns: a new Table containing the matching records
        """
        ret = self.copy_template()
        if maxrecs:
            ret.insert_many(islice(ifilter(wherefn, self.obs), 0, maxrecs))
        else:
            ret.insert_many(ifilter(wherefn, self.obs))
        return ret

    def join(self, other, attrlist=None, **kwargs):
        """
        Join the objects of one table with the objects of another, based on the given 
        matching attributes in the named arguments.  The attrlist specifies the attributes to 
        be copied from the source tables - if omitted, all attributes will be copied.  Entries 
        in the attrlist may be single attribute names, or if there are duplicate names in both
        tables, then a C{(table,attributename)} tuple can be given to disambiguate which 
        attribute is desired. A C{(table,attributename,alias)} tuple can also be passed, to 
        rename an attribute from a source table.
        
        This method may be called directly, or can be constructed using the L{join_on} method and
        the '+' operator.  Using this syntax, the join is specified using C{table.join_on("xyz")}
        to create a JoinTerm containing both table and joining attribute.  Multiple JoinTerm
        or tables can be added to construct a compound join expression.  When complete, the 
        join expression gets executed by calling the resulting join definition, 
        using C{join_expression([attrlist])}.
        
        @param other: other table to join to
        @param attrlist: list of attributes to be copied to the new joined table; if 
            none provided, all attributes of both tables will be used (taken from the first 
            object in each table)
        @type attrlist: string, or list of strings or C{(table,attribute[,alias])} tuples
            (list may contain both strings and tuples)
        @param **kwargs: attributes to join on, given as additional named arguments
            of the form C{table1attr="table2attr"}, or a dict mapping attribute names.
        @returns: a new Table containing the joined data as new DataObjects
        """
        thiscol,othercol = kwargs.items()[0]

        retname = ("(%s:%s^%s:%s)" % 
                (self.table_name, thiscol, other.table_name, othercol))
        # make sure both tables contain records to join - if not, just return empty list
        if not (self.obs and other.obs):
            return Table(retname)
        
        if isinstance(attrlist, basestring):
            attrlist = attrlist.split()
            
        # expand attrlist to full (table, name, alias) tuples
        thisnames = set(_object_attrnames(self.obs[0]))
        othernames = set(_object_attrnames(other.obs[0]))
        fullcols = []
        if attrlist is not None:
            for col in attrlist:
                if isinstance(col, tuple):
                    # assume col contains at least (table, colname), fill in alias if missing 
                    # to be same as colname
                    fullcols.append((col + (col[1],))[:3])
                else:
                    if col in thisnames:
                        fullcols.append( (self, col, col) )
                    elif col in othernames:
                        fullcols.append( (other, col, col) )
                    else:
                        pass
        else:
            fullcols = [(self,n,n) for n in thisnames]
            fullcols += [(other,n,n) for n in othernames]

        thiscols = list(ifilter(lambda o:o[0] is self, fullcols))
        othercols = list(ifilter(lambda o:o[0] is other, fullcols))

        thiscolindex = othercolindex = None
        if thiscol in self._indexes:
            thiscolindex = self._indexes[thiscol]
        if othercol in other._indexes:
            othercolindex = other._indexes[othercol]
        if not(thiscolindex and othercolindex):
            raise ValueError("can only join on indexed attributes")

        # use table with fewer keys to drive join
        if len(thiscolindex) < len(othercolindex):
            shortindex, longindex = (thiscolindex, othercolindex)
            swap = False
        else:
            shortindex, longindex = (othercolindex, thiscolindex)
            swap = True
            
        # find matching rows
        matchingrows = []
        for key,rows in shortindex.items():
            if key in longindex:
                if swap:
                    matchingrows.append( (longindex[key], rows) )
                else:
                    matchingrows.append( (rows, longindex[key]) )

        joinrows = []
        for thisrows,otherrows in matchingrows:
            for trow,orow in product(thisrows,otherrows):
                retobj = DataObject()
                for _,c,a in thiscols:
                    setattr(retobj, a, getattr(trow,c))
                for _,c,a in othercols:
                    setattr(retobj, a, getattr(orow,c))
                joinrows.append(retobj)

        ret = Table(retname)
        for tbl,collist in zip([self,other],[thiscols,othercols]):
            for _,c,a in collist:
                if c in tbl._indexes:
                    ret.create_index(a) # no unique indexes in join results
        ret.insert_many(joinrows)
        return ret

    def join_on(self, attr):
        """Creates a JoinTerm in preparation for joining with another table, to 
           indicate what attribute should be used in the join.  Only indexed attributes
           may be used in a join.
           @param attr: attribute name to join from this table (may be different
               from the attribute name in the table being joined to)
           @type attr: string
           @returns: L{JoinTerm}"""
        if attr not in self._indexes:
            raise ValueError("can only join on indexed attributes")
        return JoinTerm(self, attr)
        
    def pivot(self, attrlist):
        """Pivots the data using the given attributes, returning a L{PivotTable}.
            @param attrlist: list of attributes to be used to construct the pivot table
            @type attrlist: list of strings, or string of space-delimited attribute names
        """
        if isinstance(attrlist, basestring):
            attrlist = attrlist.split()
        if all(a in self._indexes for a in attrlist):
            return PivotTable(self,[],attrlist)
        else:
            raise ValueError("pivot can only be called using indexed attributes")

    def csv_import(self, csv_source, transforms=None):
        """Imports the contents of a CSV-formatted file into this table.
           @param csv_source: CSV file - if a string is given, the file with that name will be 
               opened, read, and closed; if a file object is given, then that object 
               will be read as-is, and left for the caller to be closed.
           @type csv_source: string or file
           @param transforms: dict of functions by attribute name; if given, each
               attribute will be transformed using the corresponding transform; if there is no 
               matching transform, the attribute will be read as a string (default); the
               transform function can also be defined as a (function, default-value) tuple; if
               there is an Exception raised by the transform function, then the attribute will
               be set to the given default value
           @type transforms: dict (optional)
        """
        close_on_exit = False
        if isinstance(csv_source, basestring):
            csv_source = open(csv_source)
            close_on_exit = True
        try:
            csvdata = csv.DictReader(csv_source)
            self.insert_many(DataObject(**s) for s in csvdata)
            if transforms:
                for attr,fn in transforms.items():
                    default = None
                    if isinstance(fn,tuple):
                        fn,default = fn
                    objfn = lambda obj : fn(getattr(obj,attr))
                    self.compute(attr, objfn, default)
        finally:
            if close_on_exit:
                csv_source.close()
            
    def csv_export(self, csv_dest, fieldnames=None):
        """Exports the contents of the table to a CSV-formatted file.
           @param csv_dest: CSV file - if a string is given, the file with that name will be 
               opened, written, and closed; if a file object is given, then that object 
               will be written as-is, and left for the caller to be closed.
           @type csv_dest: string or file
           @param fieldnames: attribute names to be exported; can be given as a single
               string with space-delimited names, or as a list of attribute names
        """
        close_on_exit = False
        if isinstance(csv_dest, basestring):
            csv_dest = open(csv_dest,'wb')
            close_on_exit = True
        try:
            if fieldnames is None:
                fieldnames = list(_object_attrnames(self.obs[0]))
            if isinstance(fieldnames, basestring):
                fieldnames = fieldnames.split()
                
            csv_dest.write(','.join(fieldnames) + '\n')
            csvout = csv.DictWriter(csv_dest, fieldnames, extrasaction='ignore')
            if hasattr(self.obs[0], "__dict__"):
                for o in self.obs:
                    csvout.writerow(o.__dict__)
            else:
                for o in self.obs:
                    row = dict(starmap(lambda obj, fld: (fld, getattr(obj, fld)),
                                       zip(repeat(o), fieldnames)))
                    csvout.writerow(row)
        finally:
            if close_on_exit:
                csv_dest.close()

    def compute(self, attrname, fn, default=None):
        """Computes a new attribute for each object in table, or replaces an
           existing attribute in each record with a computed value
           @param attrname: attribute to compute for each object
           @type attrname: string
           @param fn: function used to compute new attribute value, based on 
           other values in the object
           @type fn: function(obj) returns value
           @param default: value to use if an exception is raised while trying
           to evaluate fn
           """
        for rec in self:
            try:
                val = fn(rec)
            except Exception:
                val = default
            if isinstance(rec, DataObject):
                object.__setattr__(rec, attrname, val)
            else:
                setattr(rec, attrname, val)


class PivotTable(Table):
    """Enhanced Table containing pivot results from calling table.pivot().
    """
    def __init__(self, parent, attr_val_path, attrlist):
        """PivotTable initializer - do not create these directly, use
           L{Table.pivot}.
        """
        super(PivotTable,self).__init__()
        self._attr_path = attr_val_path[:]
        self._pivot_attrs = attrlist[:]
        self._subtable_dict = {}
        
        for k,v in parent._indexes.items():
            self._indexes[k] = v.copy_template()
        if not attr_val_path:
            self.insert_many(parent.obs)
        else:
            attr,val = attr_val_path[-1]
            self.insert_many(parent.query(**{attr:val}))
            parent._subtable_dict[val] = self

        if len(attrlist) > 0:
            this_attr = attrlist[0]
            sub_attrlist = attrlist[1:]
            ind = parent._indexes[this_attr]
            self.subtables =  [ PivotTable(self, 
                                            attr_val_path + [(this_attr,k)], 
                                            sub_attrlist) for k in sorted(ind.keys()) ]
        else:
            self.subtables = []

    def __getitem__(self,val):
        if self._subtable_dict:
            return self._subtable_dict[val]
        else:
            return super(PivotTable,self).__getitem__(val)

    def keys(self):
        return sorted(self._subtable_dict.keys())

    def items(self):
        return sorted(self._subtable_dict.items())

    def values(self):
        return self._subtable_dict.values()

    def pivot_key(self):
        """Return the set of attribute-value pairs that define the contents of this 
           table within the original source table.
        """
        return self._attr_path
        
    def pivot_key_str(self):
        """Return the pivot_key as a displayable string.
        """
        return '/'.join("%s:%s" % (attr,key) for attr,key in self._attr_path)

    def has_subtables(self):
        """Return whether this table has further subtables.
        """
        return bool(self.subtables)
    
    def dump(self, out=sys.stdout, row_fn=repr, maxrecs=-1, indent=0):
        """Dump out the contents of this table in a nested listing.
           @param out: output stream to write to
           @param row_fn: function to call to display individual rows
           @param maxrecs: number of records to show at deepest level of pivot (-1=show all)
           @param indent: current nesting level
        """
        NL = '\n'
        if indent:
            out.write("  "*indent + self.pivot_key_str())
        else:
            out.write("Pivot: %s" % ','.join(self._pivot_attrs))
        out.write(NL)
        if self.has_subtables():
            for sub in self.subtables:
                if sub:
                    sub.dump(out, row_fn, maxrecs, indent+1)
        else:
            if maxrecs >= 0:
                showslice = slice(0,maxrecs)
            else:
                showslice = slice(None,None)
            for r in self.obs[showslice]:
                out.write("  "*(indent+1) + row_fn(r) + NL)
        out.flush()
        
    def dump_counts(self, out=sys.stdout):
        """Dump out the summary counts of entries in this pivot table as a tabular listing.
           @param out: output stream to write to
        """
        if len(self._pivot_attrs) == 1:
            out.write("Pivot: %s\n" % ','.join(self._pivot_attrs))
            maxkeylen = max(len(str(k)) for k in self.keys())
            for sub in self.subtables:
                out.write("%-*.*s " % (maxkeylen,maxkeylen,sub._attr_path[-1][1]))
                out.write("%7d\n" % len(sub))
        elif len(self._pivot_attrs) == 2:
            out.write("Pivot: %s\n" % ','.join(self._pivot_attrs))
            maxkeylen = max(max(len(str(k)) for k in self.keys()),5)
            maxvallen = max(max(len(str(k)) for k in self.subtables[0].keys()),7)
            keytally = dict((k,0) for k in self.subtables[0].keys())
            out.write("%*s " % (maxkeylen,''))
            out.write(' '.join("%*.*s" % (maxvallen,maxvallen,k) for k in self.subtables[0].keys()))
            out.write('   Total\n')
            for sub in self.subtables:
                out.write("%-*.*s " % (maxkeylen,maxkeylen,sub._attr_path[-1][1]))
                for ssub in sub.subtables:
                    out.write("%*d " % (maxvallen,len(ssub)))
                    keytally[ssub._attr_path[-1][1]] += len(ssub)
                out.write("%7d\n" % len(sub))
            out.write('%-*.*s ' % (maxkeylen,maxkeylen,"Total"))
            out.write(' '.join("%*d" % (maxvallen,tally) for k,tally in sorted(keytally.items())))
            out.write(" %7d\n" % sum(tally for k,tally in keytally.items()))
        else:
            raise ValueError("can only dump summary counts for 1 or 2-attribute pivots")

    def summary_counts(self,fn=None,col=None):
        """Dump out the summary counts of this pivot table as a Table.
        """
        ret = Table()
        topattr = self._pivot_attrs[0]
        for attr in self._pivot_attrs:
            ret.create_index(attr)
        if len(self._pivot_attrs) == 1:
            for sub in self.subtables:
                subattr,subval = sub._attr_path[-1]
                if fn is None:
                    ret.insert(DataObject(**{subattr:subval, 'Count':len(sub)}))
                else:
                    attrdict[fn.__name__] = reduce(fn, (s[col] for s in sub))
        elif len(self._pivot_attrs) == 2:
            for sub in self.subtables:
                for ssub in sub.subtables:
                    attrdict = dict(ssub._attr_path)
                    if fn is None:
                        attrdict['Count'] = len(ssub)
                    else:
                        attrdict[fn.__name__] = reduce(fn, (s[col] for s in ssub))
                    ret.insert(DataObject(**attrdict))
        elif len(self._pivot_attrs) == 3:
            for sub in self.subtables:
                for ssub in sub.subtables:
                    for sssub in ssub.subtables:
                        attrdict = dict(sssub._attr_path)
                        if fn is None:
                            attrdict['Count'] = len(sssub)
                        else:
                            attrdict[fn.__name__] = reduce(fn, (s[col] for s in sssub))
                        ret.insert(DataObject(**attrdict))
        else:
            raise ValueError("can only dump summary counts for 1 or 2-attribute pivots")
        return ret

class JoinTerm(object):
    """Temporary object created while composing a join across tables using 
       L{Table.join_on} and '+' addition. JoinTerm's are usually created by 
       calling join_on on a Table object, as in::
       
           customers.join_on("id") + orders.join_on("custid")
        
       This join expression would set up the join relationship 
       equivalent to::
       
           customers.join(orders, id="custid")
           
       If tables are being joined on attributes that have the same name in 
       both tables, then a join expression could be created by adding a
       JoinTerm of one table directly to the other table::
       
           customers.join_on("custid") + orders
       
       Once the join expression is composed, the actual join is performed 
       using function call notation::

           customerorders = customers.join_on("custid") + orders
           for custord in customerorders():
               print custord

       When calling the join expression, you can optionally specify a
       list of attributes as defined in L{Table.join}.
    """
    def __init__(self, sourceTable, joinfield):
        self.sourcetable = sourceTable
        self.joinfield = joinfield
        self.jointo = None

    def __add__(self, other):
        if isinstance(other, Table):
            other = other.join_on(self.joinfield)
        if isinstance(other, JoinTerm):
            if self.jointo is None:
                if other.jointo is None:
                    self.jointo = other
                else:
                    self.jointo = other()
                return self
            else:
                if other.jointo is None:
                    return self() + other
                else:
                    return self() + other()
        raise ValueError("cannot add object of type '%s' to JoinTerm" % other.__class__.__name__)

    def __radd__(self, other):
        if isinstance(other, Table):
            return other.join_on(self.joinfield) + self
        raise ValueError("cannot add object of type '%s' to JoinTerm" % other.__class__.__name__)
            
    def __call__(self, attrs=None):
        if self.jointo:
            other = self.jointo
            if isinstance(other, Table):
                other = other.join_on(self.joinfield)
            ret = self.sourcetable.join(other.sourcetable, attrs, 
                                        **{self.joinfield : other.joinfield})
            return ret
        else:
            return self.sourcetable.query()

    def join_on(self, col):
        return self().join_on(col)
        

if __name__ == "__main__":
    
    # import json in Python 2 or 3 compatible forms
    from functools import partial
    try:
        import simplejson as json
        json_dumps = partial(json.dumps, indent='  ')
    except ImportError:
        import json
        json_dumps = partial(json.dumps, indent=2)
        

    rawdata = """\
    Phoenix:AZ:85001:KPHX
    Phoenix:AZ:85001:KPHY
    Phoenix:AZ:85001:KPHA
    Dallas:TX:75201:KDFW""".splitlines()

    # load miniDB
    stations = Table()
    #~ stations.create_index("city")
    stations.create_index("stn", unique=True)

    fields = "city state zip stn".split()
    for d in rawdata:
        ob = DataObject()
        for k,v in zip(fields, d.split(':')):
            setattr(ob,k,v.strip())
        stations.insert(ob)

    # perform some queries and deletes
    for queryargs in [
        dict(city="Phoenix"),
        dict(city="Phoenix", stn="KPHX"),
        dict(stn="KPHA", city="Phoenix"),
        dict(state="TX"),
        dict(city="New York"),
        dict(city="Phoenix", _orderby="stn"),
        dict(city="Phoenix", _orderbydesc="stn"),
        ]:
        print queryargs,
        result = stations.query(**queryargs)
        print len(result)
        for r in result: print r
        print
    #~ print stations.delete(city="Phoenix")
    #~ print stations.delete(city="Boston")
    print list(stations.query())
    print

    amfm = Table()
    amfm.create_index("stn", unique=True)
    amfm.insert(DataObject(stn="KPHY", band="AM"))
    amfm.insert(DataObject(stn="KPHX", band="FM"))
    amfm.insert(DataObject(stn="KPHA", band="FM"))
    amfm.insert(DataObject(stn="KDFW", band="FM"))
    
    try:
        amfm.insert(DataObject(stn="KPHA", band="AM"))
    except KeyError:
        print "duplicate key not allowed"

    print
    for rec in (stations.join_on("stn") + amfm.join_on("stn")
                )(["stn", "city", (amfm,"band","AMFM"), 
                   (stations,"state","st")]).query(_orderby="AMFM"):
        print repr(rec)

    print
    for rec in (stations.join_on("stn") + amfm.join_on("stn")
                )(["stn", "city", (amfm,"band"), (stations,"state","st")]):
        print json_dumps(rec.__dict__)

    print
    for rec in (stations.join_on("stn") + amfm.join_on("stn"))():
        print json_dumps(rec.__dict__)

    print
    stations.create_index("state")
    pivot = stations.pivot("state")
    pivot.dump_counts()
    
    print
    amfm.create_index("band")
    pivot = (stations.join_on("stn") + amfm)().pivot("state band")
    pivot.dump_counts()
