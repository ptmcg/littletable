#
#
# littletable.py
# 
# littletable is a simple in-memory database for ad-hoc or user-defined objects,
# supporting simple query and join operations - useful for ORM-like access
# to a collection of data objects, without dealing with SQL
#
#
# Copyright (c) 2010-2016  Paul T. McGuire
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

In addition to basic insert/remove/query/delete access to the contents of a 
Table, C{littletable} offers:
 - simple indexing for improved retrieval performance, and optional enforcing key uniqueness
 - access to objects using indexed attributes
 - simplified joins using '+' operator syntax between annotated Tables
 - the result of any query or join is a new first-class C{littletable} Table
 - pivot on one or two attributes to gather tabulated data summaries
 - easy import/export to CSV and JSON files

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
    print(catalog.by.sku["ANVIL-001"].descr)

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
    print(customers.by.id["0030"].name)

    # print all items sold by the pound
    for item in catalog.where(unitofmeas="LB"):
        print(item.sku, item.descr)

    # print all items that cost more than 10
    for item in catalog.where(lambda o : o.unitprice>10):
        print(item.sku, item.descr, item.unitprice)

    # join tables to create queryable wishlists collection
    wishlists = customers.join_on("id") + wishitems.join_on("custid") + catalog.join_on("sku")

    # print all wishlist items with price > 10
    bigticketitems = wishlists().where(lambda ob : ob.unitprice > 10)
    for item in bigticketitems:
        print(item)

    # list all wishlist items in descending order by price
    for item in wishlists().sort("unitprice desc"):
        print(item)
"""

__version__ = "0.10"
__versionTime__ = "08 Nov 2016 00:34"
__author__ = "Paul McGuire <ptmcg@users.sourceforge.net>"

import sys
from operator import attrgetter
import csv
from collections import defaultdict, deque, namedtuple
from itertools import groupby,islice,starmap,repeat

PY_2 = sys.version_info[0] == 2
PY_3 = sys.version_info[0] == 3

if PY_2:
    from itertools import ifilter as filter

try:
    from collections import OrderedDict as ODict
except ImportError:
    # best effort, just use dict, but won't preserve ordering of fields
    # in tables or output files
    ODict = dict

import re
# import json in Python 2 or 3 compatible forms
from functools import partial

try:
    import simplejson as json

    json_dumps = partial(json.dumps, indent='  ')
except ImportError:
    import json

    json_dumps = partial(json.dumps, indent=2)

_consumer = deque(maxlen=0)
do_all = _consumer.extend

try:
    from itertools import product
except ImportError:
    def product(*seqs):
        tupleseqs = [[(x,) for x in s] for s in seqs]

        def _product(*internal_seqs):
            if len(internal_seqs) == 1:
                for x in internal_seqs[0]:
                    yield x
            else:
                for x in internal_seqs[0]:
                    for p in _product(*internal_seqs[1:]):
                        yield x+p

        for pp in _product(*tupleseqs):
            yield pp

if PY_3:
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

def _to_json(obj):
    if hasattr(obj, "__dict__"):
        # normal object
        return json.dumps(obj.__dict__)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # namedtuple
        return json.dumps(ODict(zip(obj._fields, obj)))
    elif hasattr(obj, "__slots__"):
        return json.dumps(ODict((k, v) for k, v in zip(obj.__slots__, (getattr(obj, a) for a in obj.__slots__))))
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
        return '{' + ', '.join(("%r: %r" % k_v) for k_v in self.__dict__.items()) + '}'
    def __setattr__(self, attr, val):
        # make all attributes write-once
        if attr not in self.__dict__:
            super(DataObject, self).__setattr__(attr, val)
        else:
            raise AttributeError("can't set existing attribute")
    def __getitem__(self, k):
        if hasattr(self, k):
            return getattr(self, k)
        else:
            raise KeyError("object has no such attribute " + k)
    def __setitem__(self, k, v):
        if k not in self.__dict__:
            self.__dict__[k] = v
        else:
            raise KeyError("attribute already exists")
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class _ObjIndex(object):
    def __init__(self, attr):
        self.attr = attr
        self.obs = defaultdict(list)
        self.is_unique = False
    def __setitem__(self, k, v):
        self.obs[k].append(v)
    def __getitem__(self, k):
        return self.obs.get(k, [])

    def __len__(self):
        return len(self.obs)
    def __iter__(self):
        return iter(self.obs.keys())

    def keys(self):
        return sorted(filter(None, self.obs.keys()))
    def items(self):
        return self.obs.items()
    def remove(self, obj):
        try:
            k = getattr(obj, self.attr)
            self.obs[k].remove(obj)
        except (ValueError, AttributeError, KeyError):
            pass
    def __contains__(self, key):
        return key in self.obs
    def copy_template(self):
        return self.__class__(self.attr)
        
class _UniqueObjIndex(_ObjIndex):
    def __init__(self, attr, accept_none=False):
        super(_UniqueObjIndex, self).__init__(attr)
        self.obs = {}
        self.is_unique = True
        self.accept_none = accept_none
        self.none_values = []

    def __setitem__(self, k, v):
        if k is not None:
            if k not in self.obs:
                self.obs[k] = v
            else:
                raise KeyError("duplicate key value %s" % k)
        else:
            if self.accept_none:
                self.none_values.append(v)
            else:
                raise ValueError("None is not a valid index key")

    def __getitem__(self, k):
        if k is not None:
            return [self.obs.get(k)] if k in self.obs else []
        else:
            return list(self.none_values)
    def __contains__(self, k):
        if k is not None:
            return k in self.obs
        else:
            return self.accept_none and self.none_values
    def keys(self):
        return sorted(self.obs.keys()) + ([None, ] if self.none_values else [])

    def items(self):
        return ((k, [v]) for k, v in self.obs.items())

    def remove(self, obj):
        k = getattr(obj, self.attr)
        if k is not None:
            self.obs.pop(k, None)
        else:
            try:
                self.none_values.remove(obj)
            except ValueError:
                pass

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


class _UniqueObjIndexWrapper(_ObjIndexWrapper):
    def __getitem__(self, k):
        if k is not None:
            try:
                return self._index[k][0]
            except IndexError:
                raise KeyError("no such value %r in index %r" % (k, self._index.attr))
        else:
            ret = Table()
            if k in self._index:
                ret.insert_many(self._index[k])
            return ret
            
class _IndexAccessor(object):
    def __init__(self, table):
        self.table = table
        
    def __getattr__(self, attr):
        """A quick way to query for matching records using their indexed attributes. The attribute
           name is used to locate the index, and returns a wrapper on the index.  This wrapper provides
           dict-like access to the underlying records in the table, as in::
           
              employees.by.socsecnum["000-00-0000"]
              customers.by.zipcode["12345"]
        
           (C{'by'} is added as a pseudo-attribute on tables, to help indicate that the indexed attributes
           are not attributes of the table, but of items in the table.)

           The behavior differs slightly for unique and non-unique indexes:
             - if the index is unique, then retrieving a matching object, will return just the object;
               if there is no matching object, C{KeyError} is raised (making a table with a unique
               index behave very much like a Python dict)
             - if the index is non-unique, then all matching objects will be returned in a new Table,
               just as if a regular query had been performed; if no objects match the key value, an empty
               Table is returned and no exception is raised.
               
           If there is no index defined for the given attribute, then C{AttributeError} is raised.
        """
        if attr in self.table._indexes:
            ret = self.table._indexes[attr]
            if isinstance(ret, _UniqueObjIndex):
                ret = _UniqueObjIndexWrapper(ret)
            if isinstance(ret, _ObjIndex):
                ret = _ObjIndexWrapper(ret)
            return ret
        raise AttributeError("Table %r has no index %r" % (self.table.table_name, attr))


class Table(object):
    """Table is the main class in C{littletable}, for representing a collection of DataObjects or
       user-defined objects with publicly accessible attributes or properties.  Tables can be:
        - created, with an optional name, using standard Python L{C{Table() constructor}<__init__>}
        - indexed, with multiple indexes, with unique or non-unique values, see L{create_index}
        - queried, specifying values to exact match in the desired records, see L{where}
        - filtered (using L{where}), using a simple predicate function to match desired records;
          useful for selecting using inequalities or compound conditions
        - accessed directly for keyed values, using C{table.indexattribute[key]} - see L{__getattr__}
        - joined, using L{join_on} to identify attribute to be used for joining with another table, and
          L{join} or operator '+' to perform the actual join
        - pivoted, using L{pivot} to create a nested structure of sub-tables grouping objects
          by attribute values
        - grouped, using L{groupby} to create a summary table of computed values, grouped by a key 
          attribute
        - L{imported<csv_import>}/L{exported<csv_export>} to CSV-format files
       Queries and joins return their results as new Table objects, so that queries and joins can
       be easily performed as a succession of operations.
    """
    def __init__(self, table_name=''):
        """Create a new, empty Table.
           @param table_name: name for Table
           @type table_name: string (optional)
        """
        self(table_name)
        self.obs = []
        self._indexes = {}
        self._uniqueIndexes = []
        self.by = _IndexAccessor(self)
        """
        C{'by'} is added as a pseudo-attribute on tables, to provide
        dict-like access to the underlying records in the table by index key, as in::
           
            employees.by.socsecnum["000-00-0000"]
            customers.by.zipcode["12345"]
              
        The behavior differs slightly for unique and non-unique indexes:
         - if the index is unique, then retrieving a matching object, will return just the object;
           if there is no matching object, C{KeyError} is raised (making a table with a unique
           index behave very much like a Python dict)
         - if the index is non-unique, then all matching objects will be returned in a new Table,
           just as if a regular query had been performed; if no objects match the key value, an empty
           Table is returned and no exception is raised.

        If there is no index defined for the given attribute, then C{AttributeError} is raised.
        """
        
    def __len__(self):
        """Return the number of objects in the Table."""
        return len(self.obs)
        
    def __iter__(self):
        """Create an iterator over the objects in the Table."""
        return iter(self.obs)
        
    def __getitem__(self, i):
        """Provides direct indexed/sliced access to the Table's underlying list of objects."""
        if isinstance(i, slice):
            ret = self.copy_template()
            ret.insert_many(self.obs[i])
            return ret
        else:
            return self.obs[i]
    
    def __delitem__(self, i):
        if isinstance(i, int):
            delidxs = [i]
        elif isinstance(i, slice):
            obs_len = len(self.obs)
            norm_slice = slice(i.start if i.start >= 0 else i.start+obs_len, 
                               i.stop if i.stop >= 0 else i.stop+obs_len, 
                               i.step)
            delidxs = sorted(range(norm_slice.start, norm_slice.stop, norm_slice.step), reverse=True)
        else:
            raise TypeError("Table index must be int or slice")

        for idx in delidxs:
            self.pop(idx)

    def pop(self, i):
        ret = self.obs.pop(i)

        # remove from indexes
        do_all(ind.remove(ret) for attr,ind in self._indexes.items())
        
        return ret
        
    def __getattr__(self, attr):
        """(Deprecated) A quick way to query for matching records using their indexed attributes. The attribute
           name is used to locate the index, and returns a wrapper on the index.  This wrapper provides
           dict-like access to the underlying records in the table, as in::
           
              employees.socsecnum["000-00-0000"]
              customers.zipcode["12345"]
        
           (L{by} is added as a pseudo-attribute on tables, to help indicate that the indexed attributes
           are not attributes of the table, but of items in the table. Use of C{'by'} is preferred, and 
           will replace direct attribute access in a future release.)::

              employees.by.socsecnum["000-00-0000"]
              customers.by.zipcode["12345"]
              
           The behavior differs slightly for unique and non-unique indexes:
             - if the index is unique, then retrieving a matching object, will return just the object;
               if there is no matching object, C{KeyError} is raised (making a table with a unique
               index behave very much like a Python dict)
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
        raise AttributeError("Table %r has no index %r" % (self.table_name, attr))

    def __bool__(self):
        return bool(self.obs)
    
    __nonzero__ = __bool__
    
    def __add__(self, other):
        """Support UNION of 2 tables using "+" operator."""
        if isinstance(other, JoinTerm):
            # special case if added to a JoinTerm, do join, not union
            return other + self
        elif isinstance(other, Table):
            # if other is another Table, just union them
            return self.union(other)
        else:
            # assume other is a sequence of some sort, insert all elements
            return self.clone().insert_many(other)

    def __iadd__(self, other):
        """Support UNION of 2 tables using "+=" operator."""
        return self.insert_many(other)

    def union(self, other):
        return self.clone().insert_many(other.obs)

    def __call__(self, table_name=None):
        """A simple way to assign a name to a table, such as those
           dynamically created by joins and queries.
           @param table_name: name for Table
           @type table_name: string
        """
        if table_name is not None:
            self.table_name = table_name
        return self

    def copy_template(self, name=None):
        """Create empty copy of the current table, with copies of all
           index definitions.
        """
        ret = Table(self.table_name)
        ret._indexes.update(dict((k, v.copy_template()) for k, v in self._indexes.items()))
        ret(name)
        return ret

    def clone(self, name=None):
        """Create full copy of the current table, including table contents
           and index definitions.
        """
        ret = self.copy_template().insert_many(self.obs)(name)
        return ret

    def create_index(self, attr, unique=False, accept_none=False):
        """Create a new index on a given attribute.
           If C{unique} is True and records are found in the table with duplicate
           attribute values, the index is deleted and C{KeyError} is raised.

           If the table already has an index on the given attribute, then
           ValueError is raised.
           @param attr: the attribute to be used for indexed access and joins
           @type attr: string
           @param unique: flag indicating whether the indexed field values are 
               expected to be unique across table entries
           @type unique: boolean
           @param accept_none: flag indicating whether None is an acceptable
               unique key value for this attribute (always True for non-unique
               indexes, default=False for unique indexes)
           @type accept_none: boolean
        """
        if attr in self._indexes:
            raise ValueError('index %r already defined for table' % attr)
            
        if unique:
            self._indexes[attr] = _UniqueObjIndex(attr, accept_none)
            self._uniqueIndexes = [ind for ind in self._indexes.values() if ind.is_unique]
        else:
            self._indexes[attr] = _ObjIndex(attr)
            accept_none = True
        ind = self._indexes[attr]
        try:
            for obj in self.obs:
                obval = getattr(obj, attr, None)
                if obval is not None or accept_none:
                    ind[obval] = obj
                else:
                    raise KeyError("None is not an allowed key")
            return self
                    
        except KeyError:
            del self._indexes[attr]
            self._uniqueIndexes = [ind for ind in self._indexes.values() if ind.is_unique]
            raise
    
    def delete_index(self, attr):
        """Deletes an index from the Table.  Can be used to drop and rebuild an index,
           or to convert a non-unique index to a unique index, or vice versa.
           @param attr: name of an indexed attribute
           @type attr: string
        """
        if attr in self._indexes:
            del self._indexes[attr]
            self._uniqueIndexes = [ind for ind in self._indexes.values() if ind.is_unique]
        return self
            
    def insert(self, obj):
        """Insert a new object into this Table.
           @param obj: any Python object -
           Objects can be constructed using the defined DataObject type, or they can
           be any Python object that does not use the Python C{__slots__} feature; C{littletable}
           introspects the object's C{__dict__} or C{_fields} attributes to obtain join and 
           index attributes and values.
           
           If the table contains a unique index, and the record to be inserted would add
           a duplicate value for the indexed attribute, then C{KeyError} is raised, and the
           object is not inserted.
           
           If the table has no unique indexes, then it is possible to insert duplicate
           objects into the table.
           """
           
        # verify new object doesn't duplicate any existing unique index values
        unique_indexes = self._uniqueIndexes  # [ind for ind in self._indexes.values() if ind.is_unique]
        NO_SUCH_ATTR = object()
        if any((getattr(obj, ind.attr, None) is None and not ind.accept_none) or
               (getattr(obj, ind.attr, NO_SUCH_ATTR) in ind)
                for ind in unique_indexes):
            # had a problem, find which one
            for ind in unique_indexes:
                if getattr(obj, ind.attr, None) is None and not ind.accept_none:
                    raise KeyError("unique key cannot be None or blank for index %s" % ind.attr, obj)
                if getattr(obj, ind.attr) in ind:
                    raise KeyError("duplicate unique key value '%s' for index %s" % (getattr(obj, ind.attr), ind.attr), 
                                   obj)

        self.obs.append(obj)
        for attr, ind in self._indexes.items():
            obval = getattr(obj, attr)
            ind[obval] = obj
        return self
            
    def insert_many(self, it):
        """Inserts a collection of objects into the table."""
        do_all(self.insert(ob) for ob in it)
        return self

    def remove(self, ob):
        """Removes an object from the table. If object is not in the table, then
           no action is taken and no exception is raised."""
        # remove from indexes
        do_all(ind.remove(ob) for attr, ind in self._indexes.items())

        # remove from main object list
        self.obs.remove(ob)
        
        return self

    def remove_many(self, it):
        """Removes a collection of objects from the table."""
        do_all(self.remove(ob) for ob in it)
        return self

    def _query_attr_sort_fn(self, attr_val):
        """Used to order where keys by most selective key first"""
        attr, v = attr_val
        if attr in self._indexes:
            idx = self._indexes[attr]
            if v in idx:
                return len(idx[v])
            else:
                return 0
        else:
            return 1e9
        
    def where(self, wherefn=None, **kwargs):
        """
        Retrieves matching objects from the table, based on given
        named parameters.  If multiple named parameters are given, then
        only objects that satisfy all of the query criteria will be returned.
        
        @param wherefn: a method or lambda that returns a boolean result, as in::
           
           lambda ob : ob.unitprice > 10
           
        @type wherefn: callable(object) returning boolean

        @param kwargs: attributes for selecting records, given as additional 
          named arguments of the form C{attrname="attrvalue"}.

        @return: a new Table containing the matching objects
        """
        if kwargs:
            # order query criteria in ascending order of number of matching items
            # for each individual given attribute; this will minimize the number 
            # of filtering records that each subsequent attribute will have to
            # handle
            kwargs = kwargs.items()
            if len(kwargs) > 1 and len(self) > 100:
                kwargs = sorted(kwargs, key=self._query_attr_sort_fn)
                
            ret = self
            NO_SUCH_ATTR = object()
            for k, v in kwargs:
                newret = ret.copy_template()
                if k in ret._indexes:
                    newret.insert_many(ret._indexes[k][v])
                else:
                    newret.insert_many(r for r in ret.obs if getattr(r, k, NO_SUCH_ATTR) == v)
                ret = newret
                if not ret:
                    break
        else:
            ret = self.clone()

        if ret and wherefn is not None:
            newret = ret.copy_template()
            newret.insert_many(filter(wherefn, ret.obs))
            ret = newret

        return ret

    def delete(self, **kwargs):
        """Deletes matching objects from the table, based on given
           named parameters.  If multiple named parameters are given, then
           only objects that satisfy all of the query criteria will be removed.
           @param kwargs: attributes for selecting records, given as additional 
              named arguments of the form C{attrname="attrvalue"}.
           @return: the number of objects removed from the table
        """
        if not kwargs:
            return 0
        
        affected = self.where(**kwargs)
        self.remove_many(affected)
        return len(affected)
    
    def sort(self, key, reverse=False):
        """Sort Table in place, using given fields as sort key.
           @param key: if this is a string, it is a comma-separated list of field names,
              optionally followed by 'desc' to indicate descending sort instead of the 
              default ascending sort; if a list or tuple, it is a list or tuple of field names
              or field names with ' desc' appended; if it is a function, then it is the 
              function to be used as the sort key function
           @param reverse: (default=False) set to True if results should be in reverse order
           @type reverse: bool
           @return: self
        """
        if isinstance(key, (basestring, list, tuple)):
            if isinstance(key, basestring):
                attrdefs = [s.strip() for s in key.split(',')]
                # leftmost attr is the most primary sort key, so do succession of 
                # sorts from right to left
                attr_orders = [(a.split()+['asc', ])[:2] for a in attrdefs][::-1]
            else:
                # attr definitions were already resolved to a sequence by the caller
                attr_orders = key
            attrs = [attr for attr, order in attr_orders]

            # special optimization if all orders are ascending or descending
            if all(order == 'asc' for attr, order in attr_orders):
                self.obs.sort(key=attrgetter(*attrs), reverse=reverse)
            elif all(order == 'desc' for attr, order in attr_orders):
                self.obs.sort(key=attrgetter(*attrs), reverse=not reverse)
            else:
                # mix of ascending and descending sorts, have to do succession of sorts
                do_all(self.obs.sort(key=attrgetter(attr), reverse=(order == "desc"))
                       for attr, order in attr_orders)
        else:
            keyfn = key
            self.obs.sort(key=keyfn, reverse=reverse)
        return self

    def select(self, fields, **exprs):
        """
        Create a new table containing a subset of attributes, with optionally 
        newly-added fields computed from each rec in the original table.

        @param fields: list of strings, or single space-delimited string, listing attribute name to be included in the
        output
        @type fields: list, or space-delimited string
        @param exprs: one or more named callable arguments, to compute additional fields using the given function
        @type exprs: C{name=callable}, callable takes the record as an argument, and returns the new attribute value
        If a string is passed as a callable, this string will be used using string formatting, given the record
        as a source of interpolation values.  For instance, C{fullName = '%(lastName)s, %(firstName)s'}
        
        """
        if isinstance(fields, basestring):
            fields = fields.split()

        def _make_string_callable(expr):
            if isinstance(expr, basestring):
                return lambda r: expr % r
            else:
                return expr

        exprs = dict((k, _make_string_callable(v)) for k, v in exprs.items())
            
        raw_tuples = []
        for ob in self.obs:
            attrvalues = tuple(getattr(ob, fieldname, None) for fieldname in fields)
            if exprs:
                attrvalues += tuple(expr(ob) for expr in exprs.values())
            raw_tuples.append(attrvalues)
        
        all_names = tuple(fields) + tuple(exprs.keys())
        return Table().insert_many(DataObject(**dict(zip(all_names, outtuple))) for outtuple in raw_tuples)

    def format(self, *fields, **exprs):
        """
        Create a new table with all string formatted attribute values, typically in preparation for
        formatted output.
        @param fields: one or more strings, each string is an attribute name to be included in the output
        @type fields: string (multiple)
        @param exprs: one or more named string arguments, to format the given attribute with a formatting string 
        @type exprs: name=string
        """
        # select_exprs = {}
        # for f in fields:
        #     select_exprs[f] = lambda r : str(getattr,f,None)
        fields = set(fields)
        select_exprs = ODict((f, lambda r, f=f: str(getattr, f, None)) for f in fields)

        for ename, expr in exprs.items():
            if isinstance(expr, basestring):
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', expr):
                    select_exprs[ename] = lambda r: str(getattr(r, expr, None))
                else:
                    if "{}" in expr or "{0}" or "{0:" in expr:
                        select_exprs[ename] = lambda r: expr.format(r)
                    else:
                        select_exprs[ename] = lambda r: expr % getattr(r, ename, "None")
        
        return self.select(**select_exprs)

    def join(self, other, attrlist=None, auto_create_indexes=True, **kwargs):
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
        @param kwargs: attributes to join on, given as additional named arguments
            of the form C{table1attr="table2attr"}, or a dict mapping attribute names.
        @returns: a new Table containing the joined data as new DataObjects
        """
        if not kwargs:
            raise TypeError("must specify at least one join attribute as a named argument")
        thiscol, othercol = next(iter(kwargs.items()))

        retname = ("(%s:%s^%s:%s)" % (self.table_name, thiscol, other.table_name, othercol))
        # make sure both tables contain records to join - if not, just return empty list
        if not (self.obs and other.obs):
            return Table(retname)
        
        if isinstance(attrlist, basestring):
            attrlist = re.split(r'[,\s]+', attrlist)
            
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
                        fullcols.append((self, col, col))
                    elif col in othernames:
                        fullcols.append((other, col, col))
                    else:
                         raise ValueError("join attribute not found: " + col)
        else:
            fullcols = [(self, n, n) for n in thisnames]
            fullcols += [(other, n, n) for n in othernames]

        thiscols = list(filter(lambda o: o[0] is self, fullcols))
        othercols = list(filter(lambda o: o[0] is other, fullcols))

        if auto_create_indexes:
            if thiscol not in self._indexes:
                self.create_index(thiscol)
            if othercol not in other._indexes:
                other.create_index(othercol)

        if thiscol in self._indexes:
            thiscolindex = self._indexes[thiscol]
        else:
            raise ValueError("indexed attribute required for join: "+thiscol)
        if othercol in other._indexes:
            othercolindex = other._indexes[othercol]
        else:
            raise ValueError("indexed attribute required for join: "+othercol)

        # use table with fewer keys to drive join
        if len(thiscolindex) < len(othercolindex):
            shortindex, longindex = (thiscolindex, othercolindex)
            swap = False
        else:
            shortindex, longindex = (othercolindex, thiscolindex)
            swap = True
            
        # find matching rows
        matchingrows = list((longindex[key], rows) if swap else (rows, longindex[key])
                            for key, rows in shortindex.items())

        joinrows = []
        for thisrows, otherrows in matchingrows:
            for trow, orow in product(thisrows, otherrows):
                retobj = DataObject()
                do_all(setattr(retobj, a, getattr(trow, c)) for _, c, a in thiscols)
                do_all(setattr(retobj, a, getattr(orow, c)) for _, c, a in othercols if not hasattr(retobj, a))
                joinrows.append(retobj)

        ret = Table(retname)
        for tbl, collist in zip([self, other], [thiscols, othercols]):
            for _, c, a in collist:
                if c in tbl._indexes:
                    if a not in ret._indexes:
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
            return PivotTable(self, [], attrlist)
        else:
            raise ValueError("pivot can only be called using indexed attributes")

    def _import(self, source, encoding, transforms=None, reader=csv.DictReader):
        close_on_exit = False
        if isinstance(source, basestring):
            if PY_3:
                source = open(source, encoding=encoding)
            else:
                source = open(source)
            close_on_exit = True
        try:
            csvdata = reader(source)
            self.insert_many(DataObject(**s) for s in csvdata)
            if transforms:
                for attr, fn in transforms.items():
                    default = None
                    if isinstance(fn, tuple):
                        fn, default = fn
                    objfn = lambda obj: fn(getattr(obj, attr))
                    self.add_field(attr, objfn, default)
        finally:
            if close_on_exit:
                source.close()

    def csv_import(self, csv_source, encoding='UTF-8', transforms=None, **kwargs):
        """Imports the contents of a CSV-formatted file into this table.
           @param csv_source: CSV file - if a string is given, the file with that name will be
               opened, read, and closed; if a file object is given, then that object
               will be read as-is, and left for the caller to be closed.
           @type csv_source: string or file
           @param encoding: encoding to be used for reading source text if C{csv_source} is
               passed as a string filename
           @type encoding: string (default='UTF-8')
           @param transforms: dict of functions by attribute name; if given, each
               attribute will be transformed using the corresponding transform; if there is no
               matching transform, the attribute will be read as a string (default); the
               transform function can also be defined as a (function, default-value) tuple; if
               there is an Exception raised by the transform function, then the attribute will
               be set to the given default value
           @type transforms: dict (optional)
           @param kwargs: additional constructor arguments for csv C{DictReader} objects, such as C{delimiter}
               or C{fieldnames}; these are passed directly through to the csv C{DictReader} constructor
           @type kwargs: named arguments (optional)
        """
        reader_args = dict((k, v) for k, v in kwargs.items() if k not in ['encoding', 'csv_source', 'transforms'])
        reader = lambda src: csv.DictReader(src, **reader_args)
        return self._import(csv_source, encoding, transforms, reader=reader)

    def _xsv_import(self, xsv_source, encoding, transforms=None, splitstr="\t"):
        xsv_reader = lambda src: csv.DictReader(src, delimiter=splitstr)
        return self._import(xsv_source, encoding, transforms, reader=xsv_reader)

    def tsv_import(self, xsv_source, encoding="UTF-8", transforms=None):
        """Imports the contents of a tab-separated data file into this table.
           @param xsv_source: tab-separated data file - if a string is given, the file with that name will be
               opened, read, and closed; if a file object is given, then that object
               will be read as-is, and left for the caller to be closed.
           @type xsv_source: string or file
           @param transforms: dict of functions by attribute name; if given, each
               attribute will be transformed using the corresponding transform; if there is no
               matching transform, the attribute will be read as a string (default); the
               transform function can also be defined as a (function, default-value) tuple; if
               there is an Exception raised by the transform function, then the attribute will
               be set to the given default value
           @type transforms: dict (optional)
        """
        return self._xsv_import(xsv_source, encoding, transforms=transforms, splitstr="\t")

    def csv_export(self, csv_dest, fieldnames=None, encoding="UTF-8"):
        """Exports the contents of the table to a CSV-formatted file.
           @param csv_dest: CSV file - if a string is given, the file with that name will be 
               opened, written, and closed; if a file object is given, then that object 
               will be written as-is, and left for the caller to be closed.
           @type csv_dest: string or file
           @param fieldnames: attribute names to be exported; can be given as a single
               string with space-delimited names, or as a list of attribute names
           @type fieldnames: list of strings
           @param encoding: string (default="UTF-8"); if csv_dest is provided as a string
               representing an output filename, an encoding argument can be provided (Python 3 only)
           @type encoding: string
        """
        close_on_exit = False
        if isinstance(csv_dest, basestring):
            if PY_3:
                csv_dest = open(csv_dest, 'w', encoding=encoding)
            else:
                csv_dest = open(csv_dest, 'w')
            close_on_exit = True
        try:
            if fieldnames is None:
                fieldnames = list(_object_attrnames(self.obs[0]))
            if isinstance(fieldnames, basestring):
                fieldnames = fieldnames.split()
                
            csv_dest.write(','.join(fieldnames) + '\n')
            csvout = csv.DictWriter(csv_dest, fieldnames, extrasaction='ignore')
            if hasattr(self.obs[0], "__dict__"):
                do_all(csvout.writerow(o.__dict__) for o in self.obs)
            else:
                do_all(csvout.writerow(ODict(starmap(lambda obj, fld: (fld, getattr(obj, fld)),
                                       zip(repeat(o), fieldnames)))) for o in self.obs)
        finally:
            if close_on_exit:
                csv_dest.close()

    def json_import(self, source, encoding="UTF-8", transforms=None):
        """Imports the contents of a JSON data file into this table.
           @param source: JSON data file - if a string is given, the file with that name will be
               opened, read, and closed; if a file object is given, then that object
               will be read as-is, and left for the caller to be closed.
           @type source: string or file
           @param transforms: dict of functions by attribute name; if given, each
               attribute will be transformed using the corresponding transform; if there is no
               matching transform, the attribute will be read as a string (default); the
               transform function can also be defined as a (function, default-value) tuple; if
               there is an Exception raised by the transform function, then the attribute will
               be set to the given default value
           @type transforms: dict (optional)
        """
        class _JsonFileReader(object):
            def __init__(self, src):
                self.source = src
            def __iter__(self):
                current = ''
                for line in self.source:
                    if current:
                        current += ' '
                    current += line
                    try:
                        yield json.loads(current)
                        current = ''
                    except Exception:
                        pass
        return self._import(source, encoding, transforms=transforms, reader=_JsonFileReader)

    def json_export(self, dest, fieldnames=None, encoding="UTF-8"):
        """Exports the contents of the table to a JSON-formatted file.
           @param dest: output file - if a string is given, the file with that name will be 
               opened, written, and closed; if a file object is given, then that object 
               will be written as-is, and left for the caller to be closed.
           @type dest: string or file
           @param fieldnames: attribute names to be exported; can be given as a single
               string with space-delimited names, or as a list of attribute names
           @type fieldnames: list of strings
           @param encoding: string (default="UTF-8"); if csv_dest is provided as a string
               representing an output filename, an encoding argument can be provided (Python 3 only)
           @type encoding: string
        """
        close_on_exit = False
        if isinstance(dest, basestring):
            if PY_3:
                dest = open(dest, 'w', encoding=encoding)
            else:
                dest = open(dest, 'w')
            close_on_exit = True
        try:
            if isinstance(fieldnames, basestring):
                fieldnames = fieldnames.split()

            if fieldnames is None:
                do_all(dest.write(_to_json(o)+'\n') for o in self.obs)
            else:
                do_all(dest.write(json.dumps(ODict((f, getattr(o, f)) for f in fieldnames))+'\n') for o in self.obs)
        finally:
            if close_on_exit:
                dest.close()

    def add_field(self, attrname, fn, default=None):
        """Computes a new attribute for each object in table, or replaces an
           existing attribute in each record with a computed value
           @param attrname: attribute to compute for each object
           @type attrname: string
           @param fn: function used to compute new attribute value, based on 
           other values in the object, as in::
               
               lambda ob : ob.commission_pct/100.0 * ob.gross_sales
               
           @type fn: function(obj) returns value
           @param default: value to use if an exception is raised while trying
           to evaluate fn
           """
        # for rec in self:
        def _add_field_to_rec(rec_, fn_=fn, default_=default):
            try:
                val = fn_(rec_)
            except Exception:
                val = default_
            if isinstance(rec_, DataObject):
                rec_.__dict__[attrname] = val
            else:
                setattr(rec_, attrname, val)
        do_all(_add_field_to_rec(r) for r in self)
        return self

    addfield = add_field
    """(Deprecated) Legacy method to add a field to all objects in table; to be replaced by L{add_field}.
    """

    def groupby(self, keyexpr, **outexprs):
        """simple prototype of group by, with support for expressions in the group-by clause 
           and outputs
           @param keyexpr: grouping field and optional expression for computing the key value;
                if a string is passed
           @type keyexpr: string or tuple
           @param outexprs: named arguments describing one or more summary values to 
           compute per key
           @type outexprs: callable, taking a sequence of objects as input and returning
           a single summary value
           """
        if isinstance(keyexpr, basestring):
            keyattrs = keyexpr.split()
            keyfn = lambda o: tuple(getattr(o, k) for k in keyattrs)

        elif isinstance(keyexpr, tuple):
            keyattrs = (keyexpr[0],)
            keyfn = keyexpr[1]

        else:
            raise TypeError("keyexpr must be string or tuple")

        groupedobs = defaultdict(list)
        do_all(groupedobs[keyfn(ob)].append(ob) for ob in self.obs)

        tbl = Table()
        do_all(tbl.create_index(k, unique=(len(keyattrs) == 1)) for k in keyattrs)
        for key, recs in sorted(groupedobs.iteritems()):
            groupobj = DataObject(**dict(zip(keyattrs, key)))
            do_all(setattr(groupobj, subkey, expr(recs)) for subkey, expr in outexprs.items())
            tbl.insert(groupobj)
        return tbl

    def run(self):
        """(Deprecated) Returns the Table. Will be removed in a future release.
        """
        return self

    def unique(self, key=None):
        """
        Create a new table of objects,containing no duplicate values.

        @param key: (default=None) optional callable for computing a representative unique key for each
        object in the table. If None, then a key will be composed as a tuple of all the values in the object.
        @type key: callable, takes the record as an argument, and returns the key value or tuple to be used
        to represent uniqueness.
        """
        ret = self.copy_template()
        seen = set()
        for ob in self:
            if key is None:
                try:
                    ob_dict = vars(ob)
                except TypeError:
                    ob_dict = dict((k, getattr(ob, k)) for k in _object_attrnames(ob))
                reckey = tuple(sorted(ob_dict.items()))
            else:
                reckey = key(ob)
            if reckey not in seen:
                seen.add(reckey)
                ret.insert(ob)
        return ret

    def info(self):
        """
        Quick method to list informative table statistics
        :return: dict listing table information and statistics
        """
        unique_indexes = set(self._uniqueIndexes)
        return {
            'len': len(self),
            'name': self.table_name,
            'fields': list(_object_attrnames(self[0])) if self else [],
            'indexes': [(iname, self._indexes[iname] in unique_indexes) for iname in self._indexes],
        }


class PivotTable(Table):
    """Enhanced Table containing pivot results from calling table.pivot().
    """
    def __init__(self, parent, attr_val_path, attrlist):
        """PivotTable initializer - do not create these directly, use
           L{Table.pivot}.
        """
        super(PivotTable, self).__init__()
        self._attr_path = attr_val_path[:]
        self._pivot_attrs = attrlist[:]
        self._subtable_dict = {}
        
        # for k,v in parent._indexes.items():
        #     self._indexes[k] = v.copy_template()
        self._indexes.update(dict((k, v.copy_template()) for k, v in parent._indexes.items()))
        if not attr_val_path:
            self.insert_many(parent.obs)
        else:
            attr, val = attr_val_path[-1]
            self.insert_many(parent.where(**{attr: val}))
            parent._subtable_dict[val] = self

        if len(attrlist) > 0:
            this_attr = attrlist[0]
            sub_attrlist = attrlist[1:]
            ind = parent._indexes[this_attr]
            self.subtables = [PivotTable(self, attr_val_path + [(this_attr, k)], sub_attrlist)
                              for k in sorted(ind.keys())]
        else:
            self.subtables = []

    def __getitem__(self, val):
        if self._subtable_dict:
            return self._subtable_dict[val]
        else:
            return super(PivotTable, self).__getitem__(val)

    def keys(self):
        return sorted(self._subtable_dict.keys())

    def items(self):
        return sorted(self._subtable_dict.items())

    def values(self):
        return [self._subtable_dict[k] for k in self.keys()]

    def pivot_key(self):
        """Return the set of attribute-value pairs that define the contents of this 
           table within the original source table.
        """
        return self._attr_path
        
    def pivot_key_str(self):
        """Return the pivot_key as a displayable string.
        """
        return '/'.join("%s:%s" % (attr, key) for attr, key in self._attr_path)

    def has_subtables(self):
        """Return whether this table has further subtables.
        """
        return bool(self.subtables)
    
    def dump(self, out=sys.stdout, row_fn=repr, limit=-1, indent=0):
        """Dump out the contents of this table in a nested listing.
           @param out: output stream to write to
           @param row_fn: function to call to display individual rows
           @param limit: number of records to show at deepest level of pivot (-1=show all)
           @param indent: current nesting level
        """
        NL = '\n'
        if indent:
            out.write("  "*indent + self.pivot_key_str())
        else:
            out.write("Pivot: %s" % ','.join(self._pivot_attrs))
        out.write(NL)
        if self.has_subtables():
            do_all(sub.dump(out, row_fn, limit, indent+1) for sub in self.subtables if sub)
        else:
            if limit >= 0:
                showslice = slice(0, limit)
            else:
                showslice = slice(None, None)
            do_all(out.write("  "*(indent+1) + row_fn(r) + NL) for r in self.obs[showslice])
        out.flush()
        
    def dump_counts(self, out=sys.stdout, count_fn=len, colwidth=10):
        """Dump out the summary counts of entries in this pivot table as a tabular listing.
           @param out: output stream to write to
           @param count_fn: (default=len) function for computing value for each pivot cell
           @param colwidth: (default=10)
        """
        if len(self._pivot_attrs) == 1:
            out.write("Pivot: %s\n" % ','.join(self._pivot_attrs))
            maxkeylen = max(len(str(k)) for k in self.keys())
            maxvallen = colwidth
            keytally = {}
            for k, sub in self.items():
                sub_v = count_fn(sub)
                maxvallen = max(maxvallen, len(str(sub_v)))
                keytally[k] = sub_v
            for k, sub in self.items():
                out.write("%-*.*s " % (maxkeylen, maxkeylen, k))
                out.write("%*s\n" % (maxvallen, keytally[k]))
        elif len(self._pivot_attrs) == 2:
            out.write("Pivot: %s\n" % ','.join(self._pivot_attrs))
            maxkeylen = max(max(len(str(k)) for k in self.keys()), 5)
            maxvallen = max(max(len(str(k)) for k in self.subtables[0].keys()), colwidth)
            keytally = dict((k, 0) for k in self.subtables[0].keys())
            out.write("%*s " % (maxkeylen, ''))
            out.write(' '.join("%*.*s" % (maxvallen, maxvallen, k) for k in self.subtables[0].keys()))
            out.write(' %*s\n' % (maxvallen, 'Total'))
            for k, sub in self.items():
                out.write("%-*.*s " % (maxkeylen, maxkeylen, k))
                for kk, ssub in sub.items():
                    ssub_v = count_fn(ssub)
                    out.write("%*d " % (maxvallen, ssub_v))
                    keytally[kk] += ssub_v
                    maxvallen = max(maxvallen, len(str(ssub_v)))
                sub_v = count_fn(sub)
                maxvallen = max(maxvallen, len(str(sub_v)))
                out.write("%*d\n" % (maxvallen, sub_v))
            out.write('%-*.*s ' % (maxkeylen, maxkeylen, "Total"))
            out.write(' '.join("%*d" % (maxvallen, tally) for k, tally in sorted(keytally.items())))
            out.write(" %*d\n" % (maxvallen, sum(tally for k, tally in keytally.items())))
        else:
            raise ValueError("can only dump summary counts for 1 or 2-attribute pivots")

    def summary_counts(self, fn=None, col=None, summarycolname=None):
        """Dump out the summary counts of this pivot table as a Table.
        """
        if summarycolname is None:
            summarycolname = col
        ret = Table()
        # topattr = self._pivot_attrs[0]
        do_all(ret.create_index(attr) for attr in self._pivot_attrs)
        if len(self._pivot_attrs) == 1:
            for sub in self.subtables:
                subattr, subval = sub._attr_path[-1]
                attrdict = {subattr: subval}
                if fn is None:
                    attrdict['Count'] = len(sub)
                else:
                    attrdict[summarycolname] = fn(s[col] for s in sub)
                ret.insert(DataObject(**attrdict))
        elif len(self._pivot_attrs) == 2:
            for sub in self.subtables:
                for ssub in sub.subtables:
                    attrdict = dict(ssub._attr_path)
                    if fn is None:
                        attrdict['Count'] = len(ssub)
                    else:
                        attrdict[summarycolname] = fn(s[col] for s in ssub)
                    ret.insert(DataObject(**attrdict))
        elif len(self._pivot_attrs) == 3:
            for sub in self.subtables:
                for ssub in sub.subtables:
                    for sssub in ssub.subtables:
                        attrdict = dict(sssub._attr_path)
                        if fn is None:
                            attrdict['Count'] = len(sssub)
                        else:
                            attrdict[summarycolname] = fn(s[col] for s in sssub)
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
    def __init__(self, sourcetable, joinfield):
        self.sourcetable = sourcetable
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
        raise ValueError("cannot add object of type %r to JoinTerm" % other.__class__.__name__)

    def __radd__(self, other):
        if isinstance(other, Table):
            return other.join_on(self.joinfield) + self
        raise ValueError("cannot add object of type %r to JoinTerm" % other.__class__.__name__)
            
    def __call__(self, attrs=None):
        if self.jointo:
            other = self.jointo
            if isinstance(other, Table):
                other = other.join_on(self.joinfield)
            ret = self.sourcetable.join(other.sourcetable, attrs, 
                                        **{self.joinfield: other.joinfield})
            return ret
        else:
            return self.sourcetable.query()

    def join_on(self, col):
        return self().join_on(col)
        

if __name__ == "__main__":
    
    rawdata = """\
    Phoenix:AZ:85001:KPHX
    Phoenix:AZ:85001:KPHY
    Phoenix:AZ:85001:KPHA
    Dallas:TX:75201:KDFW""".splitlines()

    # load miniDB
    stations = Table()
    # stations.create_index("city")
    stations.create_index("stn", unique=True)

    data_fields = "city state zip stn".split()
    for d in rawdata:
        rec = DataObject()
        for kk, vv in zip(data_fields, d.split(':')):
            setattr(rec, kk, vv.strip())
        stations.insert(rec)

    # perform some queries and deletes
    for queryargs in [
        dict(city="Phoenix"),
        dict(city="Phoenix", stn="KPHX"),
        dict(stn="KPHA", city="Phoenix"),
        dict(state="TX"),
        dict(city="New York"),
        ]:
        print(queryargs)
        result = stations.where(**queryargs)
        print(len(result))
        for rec in result:
            print(rec)
        print('')

    # print stations.delete(city="Phoenix")
    # print stations.delete(city="Boston")
    print(list(stations.where()))
    print('')

    amfm = Table()
    amfm.create_index("stn", unique=True)
    amfm.insert(DataObject(stn="KPHY", band="AM"))
    amfm.insert(DataObject(stn="KPHX", band="FM"))
    amfm.insert(DataObject(stn="KPHA", band="FM"))
    amfm.insert(DataObject(stn="KDFW", band="FM"))
    print(amfm.by.stn["KPHY"])
    print(amfm.by.stn["KPHY"].band)
    
    try:
        amfm.insert(DataObject(stn="KPHA", band="AM"))
    except KeyError:
        print("duplicate key not allowed")

    print('')
    for rec in (stations.join_on("stn") + amfm.join_on("stn")
                )(["stn", "city", (amfm, "band", "AMFM"),
                   (stations, "state", "st")]).sort("AMFM"):
        print(repr(rec))

    print('')
    for rec in (stations.join_on("stn") + amfm.join_on("stn")
                )(["stn", "city", (amfm, "band"), (stations, "state", "st")]):
        print(json_dumps(vars(rec)))

    print('')
    for rec in (stations.join_on("stn") + amfm.join_on("stn"))():
        print(json_dumps(vars(rec)))
        
    print('')
    stations.create_index("state")
    for az_stn in stations.by.state['AZ']:
        print(az_stn)

    print('')
    pivot = stations.pivot("state")
    pivot.dump_counts()
    
    print('')
    amfm.create_index("band")
    pivot = (stations.join_on("stn") + amfm)().pivot("state band")
    pivot.dump_counts()
    
    print('')
    for rec in amfm:
        print(rec)
    print('')
    del amfm[0:-1:2]
    for i in amfm:
        print(i)

    print('')
    print(amfm.pop(-1))
    print(len(amfm))
    print(amfm.by.stn['KPHY'])
    
    