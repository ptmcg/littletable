#
#
# littletable.py
# 
# littletable is a simple in-memory database for ad-hoc or user-defined objects,
# supporting simple query and join operations - useful for ORM-like access
# to a collection of data objects, without dealing with SQL
#
#
# Copyright (c) 2010-2019  Paul T. McGuire
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
from __future__ import print_function

__doc__ = r"""

C{littletable} - a Python module to give ORM-like access to a collection of objects

The C{littletable} module provides a low-overhead, schema-less, in-memory database access to a 
collection of user objects.  C{littletable} provides a L{DataObject} class for ad hoc creation
of semi-immutable objects that can be stored in a C{littletable} L{Table}. C{Table}s can also
contain user-defined objects, using those objects' C{__dict__}, C{__slots__}, or C{_fields}
mappings to access object attributes. Table contents can thus also include namedtuples, 
SimpleNamespaces, or dataclasses.

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

    # create table of customers
    customers = Table('customers')
    customers.create_index("id", unique=True)
    customers.insert(DataObject(id="0010", name="George Jetson"))
    customers.insert(DataObject(id="0020", name="Wile E. Coyote"))
    customers.insert(DataObject(id="0030", name="Jonny Quest"))

    # create table of product catalog (load from CSV data)
    catalog_data = '''\
    sku,descr,unitofmeas,unitprice
    BRDSD-001,Bird seed,LB,3
    BBS-001,Steel BB's,LB,5
    MAGNT-001,Magnet,EA,8
    MAGLS-001,Magnifying glass,EA,12
    ANVIL-001,1000lb anvil,EA,100
    ROPE-001,1 in. heavy rope,100FT,10
    ROBOT-001,Domestic robot,EA,5000'''
    
    catalog = lt.Table("catalog")
    catalog.create_index("sku", unique=True)
    catalog.csv_import(catalog_data, transforms={'unitprice':int})

    print(catalog.by.sku["ANVIL-001"].descr)

    # create many-to-many link table of wishlist items
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

import os
import sys
import operator
import csv
import random
from collections import defaultdict, deque, namedtuple
from itertools import starmap, repeat, islice, takewhile, chain
from functools import partial
from contextlib import closing

version_info = namedtuple("version_info", "major minor micro releaseLevel serial")
__version_info__ = version_info(1, 0, 1, "final", 0)
__version__ = (
    "{}.{}.{}".format(*__version_info__[:3])
    + ("{}{}".format(__version_info__.releaseLevel[0], __version_info__.serial), "")[
        __version_info__.releaseLevel == "final"
    ]
)
__versionTime__ = "12 Sep 2020 4:34 UTC"
__author__ = "Paul McGuire <ptmcg@austin.rr.com>"

NL = os.linesep
PY_2 = sys.version_info[0] == 2
PY_3 = sys.version_info[0] == 3

if PY_2:
    from itertools import ifilter as filter
    str_strip = lambda s: type(s).strip(s)
    import urllib2
    urlopen = urllib2.urlopen
else:
    str_strip = str.strip
    import urllib.request
    urlopen = urllib.request.urlopen

try:
    from collections import OrderedDict as ODict
except ImportError:
    # try importing ordereddict backport to Py2
    try:
        from ordereddict import OrderedDict as ODict
    except ImportError:
        # best effort, just use dict, but won't preserve ordering of fields
        # in tables or output files
        ODict = dict

import re
# import json in Python 2 or 3 compatible forms
try:
    import simplejson as json
    json_dumps = partial(json.dumps, indent='  ')
except ImportError:
    import json
    json_dumps = partial(json.dumps, indent=2)

try:
    # Python 3
    from collections.abc import Mapping, Sequence
except ImportError:
    # Python 2.7
    from collections import Mapping, Sequence

_consumer = deque(maxlen=0)
do_all = _consumer.extend

try:
    from itertools import product
except ImportError:
    # Py2 emulation
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
    from io import StringIO
else:
    from StringIO import StringIO

__all__ = ["DataObject", "Table", "FixedWidthReader"]


def _object_attrnames(obj):
    if hasattr(obj, "__dict__"):
        # normal object
        return sorted(obj.__dict__.keys())
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # namedtuple
        return obj._fields
    elif hasattr(obj, "__slots__"):
        return obj.__slots__
    else:
        raise ValueError("object with unknown attributes")


def _to_dict(obj):
    if hasattr(obj, "__dict__"):
        # normal object
        return obj.__dict__
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # namedtuple
        return ODict(zip(obj._fields, obj))
    elif hasattr(obj, "__slots__"):
        return ODict((k, v) for k, v in zip(obj.__slots__, (getattr(obj, a) for a in obj.__slots__)))
    else:
        raise ValueError("object with unknown attributes")


def _to_json(obj):
    return json.dumps(_to_dict(obj))


class DataObject(object):
    """A generic semi-mutable object for storing data values in a table. Attributes
       can be set by passing in named arguments in the constructor, or by setting them
       as C{object.attribute = value}. New attributes can be added any time, but updates
       are ignored.  Table joins are returned as a Table of DataObjects."""
    def __init__(self, **kwargs):
        if kwargs:
            self.__dict__.update(kwargs)

    def __repr__(self):
        return '{' + ', '.join(("{!r}: {!r}".format(k, v)) for k, v in sorted(self.__dict__.items())) + '}'

    def __setattr__(self, attr, val):
        # make all attributes write-once
        if attr not in self.__dict__:
            super(DataObject, self).__setattr__(attr, val)
        else:
            raise AttributeError("can't set existing attribute")

    def __hasattr__(self, key):
        return key in self.__dict__

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

    def __ne__(self, other):
        return not (self == other)


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
        return sorted(filter(partial(operator.ne, None), self.obs.keys()))

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

    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

Mapping.register(_ObjIndex)


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
                raise KeyError("duplicate key value {!r}".format(k))
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
    def __init__(self, ind, table_template):
        self._index = ind
        self._table_template = table_template

    def __getattr__(self, attr):
        return getattr(self._index, attr)

    def __getitem__(self, k):
        ret = self._table_template.copy_template()
        if k in self._index:
            ret.insert_many(self._index[k])
        return ret

    def __contains__(self, k):
        return k in self._index

Mapping.register(_ObjIndexWrapper)


class _UniqueObjIndexWrapper(_ObjIndexWrapper):
    def __getitem__(self, k):
        if k is not None:
            try:
                return self._index[k][0]
            except IndexError:
                raise KeyError("no such value {!r} in index {!r}".format(k, self._index.attr))
        else:
            ret = self._table_template.copy_template()
            if k in self._index:
                ret.insert_many(self._index[k])
            return ret


class _ReadonlyObjIndexWrapper(_ObjIndexWrapper):
    def __setitem__(self, k, value):
        raise Exception("no update access to index {!r}".format(self.attr))


class _TableAttributeValueLister(object):
    class UniquableIterator(object):
        def __init__(self, seq):
            self._seq = seq
            self._iter = iter(seq)

        def __iter__(self):
            return self._iter

        def __getattr__(self, attr):
            if attr == 'unique':
                self._iter = filter(lambda x, seen=set(): x not in seen and not seen.add(x), self._iter)
                return self
            raise AttributeError("no such attribute {!r} defined".format(attr))

    def __init__(self, table, default=None):
        self.table = table
        self.default = default

    def __getattr__(self, attr):
        if attr not in self.table._indexes:
            vals = (getattr(row, attr, self.default) for row in self.table)
        else:
            table_index = self.table._indexes[attr]
            if table_index.is_unique:
                vals = table_index.keys()
            else:
                vals = chain.from_iterable(repeat(k, len(table_index[k])) for k in table_index.keys())
        return _TableAttributeValueLister.UniquableIterator(vals)


class _IndexAccessor(object):
    def __init__(self, table):
        self._table = table

    def __dir__(self):
        ret = dir(type(self)) + list(self._table._indexes)
        return ret

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
        if attr in self._table._indexes:
            ret = self._table._indexes[attr]
            if isinstance(ret, _UniqueObjIndex):
                ret = _UniqueObjIndexWrapper(ret, self._table.copy_template())
            if isinstance(ret, _ObjIndex):
                ret = _ObjIndexWrapper(ret, self._table.copy_template())
            return ret
        raise AttributeError("Table {!r} has no index {!r}".format(self._table.table_name, attr))


class _multi_iterator(object):
    def __init__(self, seqobj, encoding='utf-8'):
        if isinstance(seqobj, basestring):
            if '\n' in seqobj:
                self._iterobj = iter(StringIO(seqobj))
            elif seqobj.startswith("http"):
                if PY_3:
                    def _decoder(seq):
                        for line in seq:
                            yield line.decode(encoding)
                    self._iterobj = _decoder(urlopen(seqobj))
                else:
                    self._iterobj = urlopen(seqobj)
            else:
                if PY_3:
                    self._iterobj = open(seqobj, encoding=encoding)
                else:
                    self._iterobj = open(seqobj)
        else:
            self._iterobj = iter(seqobj)

    def __iter__(self):
        return self._iterobj

    def close(self):
        if hasattr(self._iterobj, 'close'):
            self._iterobj.close()


class FixedWidthReader(object):
    """
    Helper class to read fixed-width data and yield a sequence of dicts
    representing each row of data.

    Parameters:
    - slice_spec: a list of tuples defining each column in the input file
        Each tuple consists of:
        - data column label
        - starting column number (0-based)
        - (optional) ending column number; if omitted or None, uses the 
          starting column of the next spec tuple as the ending column 
          for this column
        - (optional) transform function: function called with the column
          string value to convert to other type, such as int, float, 
          datetime, etc.; if omitted or None, str.strip() will be used
    - src_file: a string filename or a file-like object containing the
        fixed-width data to be loaded
    """
    def __init__(self, slice_spec, src_file, encoding='utf-8'):
        def parse_spec(spec):
            ret = []
            for cur, next_ in zip(spec, spec[1:]+[("", None, None, None)]):
                label, col, endcol, fn = (cur + (None,)*2)[:4]
                if label is None:
                    continue
                if endcol is None:
                    endcol = next_[1]
                if fn is None:
                    fn = str_strip
                ret.append((label.lower(), slice(col, endcol), fn))
            return ret

        self._slices = parse_spec(slice_spec)
        self._src_file = src_file
        self._encoding = encoding

    def __iter__(self):
        with closing(_multi_iterator(self._src_file, self._encoding)) as _srciter:
            for line in _srciter:
                if not line.strip():
                    continue
                yield dict((label, fn(line[slc])) for label, slc, fn in self._slices)


def _make_comparator(cmp_fn):
    """
    Internal function to help define Table.le, Table.lt, etc.
    """
    def comparator_with_value(value):
        def _Table_comparator_fn(attr):
            return lambda table_rec: cmp_fn(getattr(table_rec, attr), value)
        return _Table_comparator_fn
    return comparator_with_value


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
    lt = staticmethod(_make_comparator(operator.lt))
    le = staticmethod(_make_comparator(operator.le))
    gt = staticmethod(_make_comparator(operator.gt))
    ge = staticmethod(_make_comparator(operator.ge))
    ne = staticmethod(_make_comparator(operator.ne))
    eq = staticmethod(_make_comparator(operator.eq))

    def __init__(self, table_name=''):
        """Create a new, empty Table.
           @param table_name: name for Table
           @type table_name: string (optional)
        """
        self(table_name)
        self.obs = []
        self._indexes = {}
        self._uniqueIndexes = []
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

        C{'all'} is added as a pseudo-attribute to tables, to provide access to the
        values of a particular table column as a sequence. This is useful if passing the
        values on to another function that works with sequences of values.

            sum(customers.by.zipcode["12345"].all.order_total)

        """

    @property
    def all(self):
        return _TableAttributeValueLister(self)

    @property
    def by(self):
        return _IndexAccessor(self)

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
            delidxs = sorted(list(range(*i.indices(obs_len))), reverse=True)
        else:
            raise TypeError("Table index must be int or slice")

        for idx in delidxs:
            self.pop(idx)

    def pop(self, i):
        ret = self.obs.pop(i)

        # remove from indexes
        do_all(ind.remove(ret) for attr, ind in self._indexes.items())
        
        return ret

    def __bool__(self):
        return bool(self.obs)

    # Py2 compat
    __nonzero__ = __bool__

    def __reversed__(self):
        return reversed(self.obs)

    def __contains__(self, item):
        return item in self.obs

    def index(self, item):
        return self.obs.index(item)

    def count(self, item):
        return self.obs.count(item)

    def __add__(self, other):
        """Support UNION of 2 tables using "+" operator."""
        if isinstance(other, _JoinTerm):
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
            raise ValueError('index {!r} already defined for table'.format(attr))

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
            
    def get_index(self, attr):
        return _ReadonlyObjIndexWrapper(self._indexes[attr], self.copy_template())

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
        return self.insert_many([obj])
            
    def insert_many(self, it):
        """Inserts a collection of objects into the table."""
        unique_indexes = self._uniqueIndexes
        NO_SUCH_ATTR = object()
        new_objs = it
        if unique_indexes:
            new_objs = list(new_objs)
            for ind in unique_indexes:
                ind_attr = ind.attr
                new_keys = dict((getattr(obj, ind_attr, NO_SUCH_ATTR), obj) for obj in new_objs)
                if not ind.accept_none and (None in new_keys or NO_SUCH_ATTR in new_keys):
                    raise KeyError("unique key cannot be None or blank for index {}".format(ind_attr),
                                   [ob for ob in new_objs if getattr(ob, ind_attr, NO_SUCH_ATTR) is None])
                if len(new_keys) < len(new_objs):
                    raise KeyError("given sequence contains duplicate keys for index {!r}".format(ind_attr))
                for key in new_keys:
                    if key in ind:
                        obj = new_keys[key]
                        raise KeyError("duplicate unique key value {!r} for index {!r}".format(getattr(obj, ind_attr),
                                                                                               ind_attr),
                                       new_keys[key])

        if self._indexes:
            for obj in new_objs:
                self.obs.append(obj)
                for attr, ind in self._indexes.items():
                    obval = getattr(obj, attr)
                    ind[obval] = obj
        else:
            self.obs.extend(new_objs)

        return self

    def remove(self, ob):
        """Removes an object from the table. If object is not in the table, then
           no action is taken and no exception is raised."""
        return self.remove_many([ob])

    def remove_many(self, it):
        """Removes a collection of objects from the table."""
        # find indicies of objects in iterable
        to_be_deleted = list(it)
        del_indices = []
        for i, ob in enumerate(self.obs):
            try:
                tbd_index = to_be_deleted.index(ob)
            except ValueError:
                continue
            else:
                del_indices.append(i)
                to_be_deleted.pop(tbd_index)

            # quit early if we have found them all
            if not to_be_deleted:
                break

        for i in sorted(del_indices, reverse=True):
            self.pop(i)

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
          named arguments of the form C{attrname="value"}, or C{Table.le(value)}
          using any of the methods C{Table.le}, C{Table.lt}, C{Table.ge}, C{Table.gt},
          C{Table.ne}, or C{Table.eq}, corresponding to the same methods defined
          in the stdlib C{operator} module.

        @return: a new Table containing the matching objects
        """
        if kwargs:
            # order query criteria in ascending order of number of matching items
            # for each individual given attribute; this will minimize the number 
            # of filtering records that each subsequent attribute will have to
            # handle
            kwargs = list(kwargs.items())
            if len(kwargs) > 1 and len(self) > 100:
                kwargs = sorted(kwargs, key=self._query_attr_sort_fn)
                
            ret = self
            NO_SUCH_ATTR = object()
            for k, v in kwargs:
                if callable(v) and v.__name__ == '_Table_comparator_fn':
                    wherefn_k = v(k)
                    newret = ret.where(wherefn_k)
                else:
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
              named arguments of the form C{attrname="attrvalue"}, or 
              C{attrname=Table.lt(value)} (see doc for L{Table.where}).
           @return: the number of objects removed from the table
        """
        if not kwargs:
            return 0
        
        affected = self.where(**kwargs)
        self.remove_many(affected)
        return len(affected)

    def shuffle(self):
        """
        In-place random shuffle of the records in the table.
        """
        random.shuffle(self.obs)
        return self

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
                attr_orders = [(a.split()+['asc', ])[:2] for a in attrdefs]
            else:
                # attr definitions were already resolved to a sequence by the caller
                if isinstance(key[0], basestring):
                    attr_orders = [(a.split()+['asc', ])[:2] for a in key]
                else:
                    attr_orders = key
            attrs = [attr for attr, order in attr_orders]

            # special optimization if all orders are ascending or descending
            if all(order == 'asc' for attr, order in attr_orders):
                self.obs.sort(key=operator.attrgetter(*attrs), reverse=reverse)
            elif all(order == 'desc' for attr, order in attr_orders):
                self.obs.sort(key=operator.attrgetter(*attrs), reverse=not reverse)
            else:
                # mix of ascending and descending sorts, have to do succession of sorts
                # leftmost attr is the most primary sort key, so reverse attr_orders to do
                # succession of sorts from right to left
                do_all(self.obs.sort(key=operator.attrgetter(attr), reverse=(order == "desc"))
                       for attr, order in reversed(attr_orders))
        else:
            # sorting given a sort key function
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
        fields = self._parse_fields_string(fields)

        def _make_string_callable(expr):
            if isinstance(expr, basestring):
                return lambda r: expr.format(r) if not isinstance(r, (list, tuple)) else expr.format(*r)
            else:
                return expr

        exprs = dict((k, _make_string_callable(v)) for k, v in exprs.items())
            
        raw_tuples = []
        for ob in self.obs:
            attrvalues = tuple(getattr(ob, field_name, None) for field_name in fields)
            if exprs:
                attrvalues += tuple(expr(ob) for expr in exprs.values())
            raw_tuples.append(attrvalues)
        
        all_names = tuple(fields) + tuple(exprs.keys())
        ret = Table()
        ret._indexes.update(dict((k, v.copy_template()) for k, v in self._indexes.items() if k in all_names))
        return ret().insert_many(DataObject(**dict(zip(all_names, out_tuple))) for out_tuple in raw_tuples)

    def formatted_table(self, *fields, **exprs):
        """
        Create a new table with all string formatted attribute values, typically in preparation for
        formatted output.
        @param fields: one or more strings, each string is an attribute name to be included in the output
        @type fields: string (multiple)
        @param exprs: one or more named string arguments, to format the given attribute with a formatting string 
        @type exprs: name=string
        """
        select_exprs = ODict()
        for fld in fields:
            if fld not in select_exprs:
                select_exprs[fld] = lambda r, f=fld: str(getattr(r, f, "None"))

        for ename, expr in exprs.items():
            if isinstance(expr, basestring):
                if re.match(r'[a-zA-Z_][a-zA-Z0-9_]*$', expr):
                    select_exprs[ename] = lambda r: str(getattr(r, expr, "None"))
                else:
                    if "{}" in expr or "{0}" or "{0:" in expr:
                        select_exprs[ename] = lambda r: expr.format(r)
                    else:
                        select_exprs[ename] = lambda r: expr.format(getattr(r, ename, "None"))
        
        return self.select(**select_exprs)

    def format(self, fmt):
        """
        Generates a list of strings, one for each row in the table, using the input string
        as a format template for printing out a single row.
        """
        for line in self:
            yield fmt.format(**_to_dict(line))

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

        retname = ("({}:{}^{}:{})".format(self.table_name, thiscol, other.table_name, othercol))
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
                        ret.create_index(a)  # no unique indexes in join results
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
        return _JoinTerm(self, attr)
        
    def pivot(self, attrlist):
        """Pivots the data using the given attributes, returning a L{PivotTable}.
            @param attrlist: list of attributes to be used to construct the pivot table
            @type attrlist: list of strings, or string of space-delimited attribute names
        """
        if isinstance(attrlist, basestring):
            attrlist = attrlist.split()
        if all(a in self._indexes for a in attrlist):
            return _PivotTable(self, [], attrlist)
        else:
            raise ValueError("pivot can only be called using indexed attributes")

    def _import(self, source, encoding, transforms=None, reader=csv.DictReader, row_class=DataObject, limit=None):
        with closing(_multi_iterator(source, encoding)) as _srciter:
            csvdata = reader(_srciter)

            if limit is not None:
                def limiter(n, iter):
                    for i, obj in enumerate(iter, start=1):
                        if i > n:
                            break
                        yield obj
                csvdata = limiter(limit, csvdata)

            if transforms:
                def slices(seq, slice_size=128):
                    seq_iter = iter(seq)
                    while True:
                        yield islice(seq_iter, 0, slice_size)
                        try:
                            yield [next(seq_iter)]
                        except StopIteration:
                            break

                if hasattr(row_class, '__dict__') or hasattr(row_class, '_fields'):
                    make_row = lambda do, cls=row_class, vars=vars: cls(**vars(do))
                else:
                    def make_row(do, cls=row_class, cls_slots=row_class.__slots__, getattr=getattr):
                        return cls(*(getattr(do, attr, None) for attr in cls_slots))

                for slc in slices(csvdata):
                    scratch = Table().insert_many(DataObject(**s) for s in slc)
                    if not scratch:
                        continue
                    for attr, fn in transforms.items():
                        default = None
                        if isinstance(fn, tuple):
                            fn, default = fn
                        objfn = lambda obj: fn(getattr(obj, attr))
                        scratch.add_field(attr, objfn, default)
                    if row_class is DataObject:
                        self.insert_many(scratch)
                    else:
                        self.insert_many(make_row(rec) for rec in scratch)
            else:
                self.insert_many(row_class(**s) for s in csvdata)
        return self

    def csv_import(self, csv_source, encoding='utf-8', transforms=None, row_class=DataObject, limit=None, **kwargs):
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
           @param row_class: class to construct for each imported row when populating table (default=DataObject)
           @type row_class: type
           @param limit: number of records to import
           @type limit: int (optional)
           @param kwargs: additional constructor arguments for csv C{DictReader} objects, such as C{delimiter}
               or C{fieldnames}; these are passed directly through to the csv C{DictReader} constructor
           @type kwargs: named arguments (optional)
        """
        reader_args = dict((k, v) for k, v in kwargs.items() if k not in ['encoding',
                                                                          'csv_source',
                                                                          'transforms',
                                                                          'row_class',
                                                                          'limit',
                                                                          ])
        reader = lambda src: csv.DictReader(src, **reader_args)
        return self._import(csv_source, encoding, transforms, reader=reader, row_class=row_class, limit=limit)

    def _xsv_import(self, xsv_source, encoding, transforms=None, row_class=DataObject, limit=None, **kwargs):
        reader_args = dict((k, v) for k, v in kwargs.items() if k not in ['encoding',
                                                                          'xsv_source',
                                                                          'transforms',
                                                                          'row_class',
                                                                          'limit',
                                                                          ])
        xsv_reader = lambda src: csv.DictReader(src, **reader_args)
        return self._import(xsv_source, encoding, transforms, reader=xsv_reader, row_class=row_class, limit=limit)

    def tsv_import(self, xsv_source, encoding="UTF-8", transforms=None, row_class=DataObject, limit=None, **kwargs):
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
           @param row_class: class to construct for each imported row when populating table (default=DataObject)
           @type row_class: type
           @param limit: number of records to import
           @type limit: int (optional)
        """
        return self._xsv_import(xsv_source,
                                encoding,
                                transforms=transforms,
                                delimiter="\t",
                                row_class=row_class,
                                limit=limit,
                                **kwargs)

    def csv_export(self, csv_dest, fieldnames=None, encoding="UTF-8", **kwargs):
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
           @param kwargs: additional keyword args to pass through to csv.DictWriter
           @type kwargs: named arguments (optional)
        """
        writer_args = dict((k, v) for k, v in kwargs.items() if k not in ['encoding',
                                                                          'csv_dest',
                                                                          'fieldnames',
                                                                          ])
        close_on_exit = False
        if isinstance(csv_dest, basestring):
            if PY_3:
                csv_dest = open(csv_dest, 'w', newline='', encoding=encoding)
            else:
                csv_dest = open(csv_dest, 'wb')
            close_on_exit = True
        try:
            if fieldnames is None:
                fieldnames = list(_object_attrnames(self.obs[0])) if self.obs else list(self._indexes.keys())
            if isinstance(fieldnames, basestring):
                fieldnames = fieldnames.split()

            csv_dest.write(','.join(fieldnames) + NL)
            csvout = csv.DictWriter(csv_dest, fieldnames, extrasaction='ignore', lineterminator=NL, **writer_args)
            if self.obs and hasattr(self.obs[0], "__dict__"):
                csvout.writerows(o.__dict__ for o in self.obs)
            else:
                do_all(csvout.writerow(ODict(starmap(lambda obj, fld: (fld, getattr(obj, fld)),
                                                     zip(repeat(o), fieldnames)))) for o in self.obs)
        finally:
            if close_on_exit:
                csv_dest.close()

    def json_import(self, source, encoding="UTF-8", transforms=None, row_class=DataObject):
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
           @param row_class: class to construct for each imported row when populating table (default=DataObject)
           @type row_class: type
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

        return self._import(source, encoding, transforms=transforms, reader=_JsonFileReader, row_class=row_class)

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
        try:
            do_all(_add_field_to_rec(r) for r in self)
        except AttributeError:
            raise AttributeError("cannot add/modify attribute {!r} in table records".format(attrname))
        return self

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

        grouped_obs = defaultdict(list)
        do_all(grouped_obs[keyfn(ob)].append(ob) for ob in self.obs)

        tbl = Table()
        do_all(tbl.create_index(k, unique=(len(keyattrs) == 1)) for k in keyattrs)
        for key, recs in sorted(grouped_obs.items()):
            group_obj = DataObject(**dict(zip(keyattrs, key)))
            do_all(setattr(group_obj, subkey, expr(recs)) for subkey, expr in outexprs.items())
            tbl.insert(group_obj)
        return tbl

    def unique(self, key=None):
        """
        Create a new table of objects,containing no duplicate values.

        @param key: (default=None) optional callable for computing a representative unique key for each
        object in the table. If None, then a key will be composed as a tuple of all the values in the object.
        @type key: callable, takes the record as an argument, and returns the key value or tuple to be used
        to represent uniqueness.
        """
        if isinstance(key, basestring):
            key = lambda r, attr=key: getattr(r, attr, None)
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
            'indexes': [(idx_name, self._indexes[idx_name] in unique_indexes) for idx_name in self._indexes],
        }

    def stats(self, field_names, by_field=True):
        accum = dict((name, [0, 0, 0, 1e300, -1e300]) for name in field_names)
        for rec in self:
            for name in field_names:
                value = getattr(rec, name, None)
                if value is not None:
                    acc = accum[name]
                    acc[0] += 1
                    acc[1] += value
                    acc[2] += value*value
                    if value < acc[3]:
                        acc[3] = value
                    if value > acc[4]:
                        acc[4] = value

        ret = Table()
        stats = [
            ('count', lambda x: x[0]),
            ('min', lambda x: x[3]),
            ('max', lambda x: x[4]),
            ('mean', lambda x: x[1] / x[0] if x[0] != 0 else None),
            ('variance', lambda x: (x[2] - x[1]**2/x[0]) / x[0] if x[0] != 0 else None),
            ('std_dev', lambda x: (x[0]*x[2] - x[1]*x[1])**0.5 / x[0] if x[0] != 0 else None),
        ]
        if by_field:
            ret.insert_many(DataObject(name=fname,
                                       **dict((stat_name, stat_fn(accum[fname]))
                                              for stat_name, stat_fn in stats))
                            for fname in field_names)
        else:
            ret.insert_many(DataObject(stat=stat_name,
                                       **dict((fname, stat_fn(accum[fname]))
                                              for fname in field_names))
                            for stat_name, stat_fn in stats)
        return ret

    def _parse_fields_string(self, field_names):
        """
        Convert raw string or list of names to actual column names:
        - names starting with '-' indicate to suppress that field
        - '*' means include all other field names
        - if no fields are specifically included, then all fields are used
        :param field_names: str or list
        :return: expanded list of field names
        """
        if isinstance(field_names, basestring):
            field_names = field_names.split()
        if not self.obs:
            return field_names

        suppress_names = [nm[1:] for nm in field_names if nm.startswith('-')]
        field_names = [nm for nm in field_names if not nm.startswith('-')]
        if not field_names:
            field_names = ['*']
        if '*' in field_names:
            if self:
                star_fields = [name for name in _object_attrnames(self[0]) if name not in field_names]
            else:
                # no records to look at, just use names of any defined indexes
                star_fields = list(self._indexes.keys())
            fn_iter = iter(field_names)
            field_names = list(takewhile(lambda x: x != '*', fn_iter)) + star_fields + list(fn_iter)
        field_names = [nm for nm in field_names if nm not in suppress_names]
        return field_names

    def as_html(self, fields='*'):
        """
        Output the table as a rudimentary HTML table.
        @param fields: fields in the table to be shown in the table
                       - listing '*' as a field will add all unnamed fields
                       - starting a field name with '-' will suppress that name
        @type fields: list of strings or a single space-delimited string
        @return: string of generated HTML representing the selected table row attributes
        """
        fields = self._parse_fields_string(fields)

        def td_value(v):
            return '<td><div align="{}">{}</div></td>'.format(('left', 'right')[isinstance(v, (int, float))], str(v))

        def row_to_tr(r):
            return "<tr>" + "".join(td_value(getattr(r, fld)) for fld in fields) + "</tr>\n"

        ret = ""
        ret += "<table>\n"
        ret += "<tr>" + "".join(map('<th><div align="center">{}</div></th>'.format, fields)) + "</tr>\n"
        ret += "".join(map(row_to_tr, self))
        ret += "</table>"
        return ret

Sequence.register(Table)


class _PivotTable(Table):
    """Enhanced Table containing pivot results from calling table.pivot().
    """
    def __init__(self, parent, attr_val_path, attrlist):
        """PivotTable initializer - do not create these directly, use
           L{Table.pivot}.
        """
        super(_PivotTable, self).__init__()
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
            self.subtables = [_PivotTable(self, attr_val_path + [(this_attr, k)], sub_attrlist)
                              for k in sorted(ind.keys())]
        else:
            self.subtables = []

    def __getitem__(self, val):
        if self._subtable_dict:
            return self._subtable_dict[val]
        else:
            return super(_PivotTable, self).__getitem__(val)

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
        return '/'.join("{}:{}".format(attr, key) for attr, key in self._attr_path)

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
        if indent:
            out.write("  "*indent + self.pivot_key_str())
        else:
            out.write("Pivot: {}".format(','.join(self._pivot_attrs)))
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
            out.write("Pivot: {}\n".format(','.join(self._pivot_attrs)))
            maxkeylen = max(len(str(k)) for k in self.keys())
            maxvallen = colwidth
            keytally = {}
            for k, sub in self.items():
                sub_v = count_fn(sub)
                maxvallen = max(maxvallen, len(str(sub_v)))
                keytally[k] = sub_v
            for k, sub in self.items():
                out.write("{:<{}.{}s} ".format(str(k), maxkeylen, maxkeylen))
                out.write("{:{}}\n".format(keytally[k], maxvallen))
        elif len(self._pivot_attrs) == 2:
            out.write("Pivot: {}\n".format(','.join(self._pivot_attrs)))
            maxkeylen = max(max(len(str(k)) for k in self.keys()), 5)
            maxvallen = max(max(len(str(k)) for k in self.subtables[0].keys()), colwidth)
            keytally = dict((k, 0) for k in self.subtables[0].keys())
            out.write("{:{}s} ".format("", maxkeylen))
            out.write(' '.join("{:{}.{}s}".format(str(k), maxvallen, maxvallen)
                               for k in self.subtables[0].keys()))
            out.write(' {:{}s}\n'.format("Total", maxvallen))
            for k, sub in self.items():
                out.write("{:<{}.{}s} ".format(str(k), maxkeylen, maxkeylen))
                for kk, ssub in sub.items():
                    ssub_v = count_fn(ssub)
                    out.write("{:{}d} ".format(ssub_v, maxvallen))
                    keytally[kk] += ssub_v
                    maxvallen = max(maxvallen, len(str(ssub_v)))
                sub_v = count_fn(sub)
                maxvallen = max(maxvallen, len(str(sub_v)))
                out.write("{:{}d}\n".format(sub_v, maxvallen))
            out.write('{:{}.{}s} '.format("Total", maxkeylen, maxkeylen))
            out.write(' '.join("{:{}d}".format(tally, maxvallen) for k, tally in sorted(keytally.items())))
            out.write(" {:{}d}\n".format(sum(tally for k, tally in keytally.items()), maxvallen))
        else:
            raise ValueError("can only dump summary counts for 1 or 2-attribute pivots")

    def as_table(self, fn=None, col=None, col_label=None):
        """Dump out the summary counts of this pivot table as a Table.
        """
        if col_label is None:
            col_label = col
        if fn is None:
            fn = len
            if col_label is None:
                col_label = 'count'
        ret = Table()

        do_all(ret.create_index(attr) for attr in self._pivot_attrs)
        if len(self._pivot_attrs) == 1:
            for sub in self.subtables:
                subattr, subval = sub._attr_path[-1]
                attrdict = {subattr: subval}
                if col is None or fn is len:
                    attrdict[col_label] = fn(sub)
                else:
                    attrdict[col_label] = fn([s[col] for s in sub])
                ret.insert(DataObject(**attrdict))
        elif len(self._pivot_attrs) == 2:
            for sub in self.subtables:
                for ssub in sub.subtables:
                    attrdict = dict(ssub._attr_path)
                    if col is None or fn is len:
                        attrdict[col_label] = fn(ssub)
                    else:
                        attrdict[col_label] = fn([s[col] for s in ssub])
                    ret.insert(DataObject(**attrdict))
        elif len(self._pivot_attrs) == 3:
            for sub in self.subtables:
                for ssub in sub.subtables:
                    for sssub in ssub.subtables:
                        attrdict = dict(sssub._attr_path)
                        if col is None or fn is len:
                            attrdict[col_label] = fn(sssub)
                        else:
                            attrdict[col_label] = fn([s[col] for s in sssub])
                        ret.insert(DataObject(**attrdict))
        else:
            raise ValueError("can only dump summary counts for 1 or 2-attribute pivots")
        return ret
    summary_counts = as_table

    def summarize(self, count_fn=len, col_label=None):
        if col_label is None:
            if len(self._pivot_attrs) == 1:
                col_label = self._pivot_attrs[0]
            else:
                col_label = 'value'
        return _PivotTableSummary(self, self._pivot_attrs, count_fn, col_label)


class _PivotTableSummary(object):
    def __init__(self, pivot_table, pivot_attrs, count_fn=len, col_label=None):
        self._pt = pivot_table
        self._pivot_attrs = pivot_attrs
        self._fn = count_fn
        self._label = col_label

    def as_html(self, *args, **kwargs):
        if len(self._pivot_attrs) == 1:
            col = self._pivot_attrs[0]
            col_label = self._label
            data = Table().insert_many(DataObject(**{col: k, col_label: self._fn(sub)}) for k, sub in self._pt.items())
            return data.as_html((col, col_label))

        elif len(self._pivot_attrs) == 2:
            def td_value(v):
                return '<td><div align="{}">{}</div></td>'.format(('left','right')[isinstance(v, (int, float))], str(v))
            def row_to_tr(r):
                return "<tr>" + "".join(td_value(fld) for fld in r) + "</tr>\n"

            ret = ""
            ret += "<table>\n"

            keytally = dict((k, 0) for k in self._pt.subtables[0].keys())
            hdgs = sorted(keytally)
            ret += ("<tr><th/>"
                    + "".join(map('<th><div align="center">{}</div></th>'.format, hdgs))
                    + '<th><div align="center">Total</div></th></tr>\n')
            for k, sub in self._pt.items():
                row = [k]
                ssub_v_accum = 0
                for kk, ssub in sub.items():
                    ssub_v = self._fn(ssub)
                    row.append(ssub_v)
                    keytally[kk] += ssub_v
                    ssub_v_accum += ssub_v
                sub_v = ssub_v_accum  # count_fn(sub)
                row.append(sub_v)
                ret += row_to_tr(row)
            row = ['Total']
            row.extend(v for k, v in sorted(keytally.items()))
            row.append(sum(keytally.values()))
            ret += row_to_tr(row)

            ret += "</table>"
            return ret

        else:  # if len(self._pivot_attrs) >= 3:
            raise Exception("no HTML output format for 3-attribute pivot tables at this time")


class _JoinTerm(object):
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
    def __init__(self, source_table, join_field):
        self.source_table = source_table
        self.join_field = join_field
        self.join_to = None

    def __add__(self, other):
        if isinstance(other, Table):
            other = other.join_on(self.join_field)
        if isinstance(other, _JoinTerm):
            if self.join_to is None:
                if other.join_to is None:
                    self.join_to = other
                else:
                    self.join_to = other()
                return self
            else:
                if other.join_to is None:
                    return self() + other
                else:
                    return self() + other()
        raise ValueError("cannot add object of type {!r} to JoinTerm".format(other.__class__.__name__))

    def __radd__(self, other):
        if isinstance(other, Table):
            return other.join_on(self.join_field) + self
        raise ValueError("cannot add object of type {!r} to JoinTerm".format(other.__class__.__name__))
            
    def __call__(self, attrs=None):
        if self.join_to:
            other = self.join_to
            if isinstance(other, Table):
                other = other.join_on(self.join_field)
            ret = self.source_table.join(other.source_table, attrs,
                                         **{self.join_field: other.join_field})
            return ret
        else:
            return self.source_table.query()

    def join_on(self, col):
        return self().join_on(col)
        

if __name__ == "__main__":
    import textwrap
    rawdata = textwrap.dedent("""\
    Phoenix:AZ:85001:KPHX
    Phoenix:AZ:85001:KPHY
    Phoenix:AZ:85001:KPHA
    Dallas:TX:75201:KDFW""")

    # load miniDB
    stations = Table().csv_import(rawdata, delimiter=':', fieldnames=['city', 'state', 'zip', 'stn'])
    # stations.create_index("city")
    stations.create_index("stn", unique=True)

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
        print()

    # print stations.delete(city="Phoenix")
    # print stations.delete(city="Boston")
    print(list(stations.where()))
    print()

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

    print()
    for rec in (stations.join_on("stn") + amfm.join_on("stn")
                )(["stn", "city", (amfm, "band", "AMFM"),
                   (stations, "state", "st")]).sort("AMFM"):
        print(repr(rec))

    print()
    for rec in (stations.join_on("stn") + amfm.join_on("stn")
                )(["stn", "city", (amfm, "band"), (stations, "state", "st")]):
        print(json_dumps(vars(rec)))

    print()
    for rec in (stations.join_on("stn") + amfm.join_on("stn"))():
        print(json_dumps(vars(rec)))

    print()
    stations.create_index("state")
    for az_stn in stations.by.state['AZ']:
        print(az_stn)

    print()
    pivot = stations.pivot("state")
    pivot.dump_counts()

    print()
    amfm.create_index("band")
    pivot = (stations.join_on("stn") + amfm)().pivot("state band")
    pivot.dump_counts()
    
    print()
    for rec in amfm:
        print(rec)
    print()

    print(list(amfm.all.stn))
    print(list(amfm.all.band))
    print(list(amfm.unique('band').all.band))
    print(list(amfm.all.band.unique))
    print()

    del amfm[0:-1:2]
    for stn in amfm:
        print(stn)
    
    print()
    print(amfm.pop(-1))
    print(len(amfm))
    print(amfm.by.stn['KPHX'])
    try:
        print(amfm.by.stn['KPHY'])
    except KeyError:
        print("no station 'KPHY' in table")

    print(list(stations.all.stn))

    # do some simple stats with common ML data set
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    transforms = dict.fromkeys(['petal-length', 'petal-width', 'sepal-length', 'sepal-width'], float)
    iris_table = Table('iris').csv_import(url, fieldnames=names, transforms=transforms)

    print(iris_table.info())
    for rec in iris_table[:5]:
        print(rec)

    stats = iris_table.stats(['petal-length', 'petal-width', 'sepal-length', 'sepal-width'])
    for rec in stats:
        print(rec)
