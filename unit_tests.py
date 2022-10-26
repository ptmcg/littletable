#
# unit_tests.py
#
# unit tests for littletable library
#
import ast
from collections import namedtuple
import copy
import io
import itertools
import json
from operator import attrgetter
import os
import sys
import textwrap
from types import SimpleNamespace
import unittest
from typing import Optional, Union, NamedTuple

python_version = sys.version_info[:2]

import littletable as lt

SKIP_CSV_IMPORT_USING_URL_TESTS = os.environ.get("SKIP_CSV_IMPORT_USING_URL_TESTS", "0") == "1"

try:
    import dataclasses
except ImportError:
    print("dataclasses tests disabled")
    # pre Py3.7 (or 3.6 with backported dataclasses)
    dataclasses = None
else:
    @dataclasses.dataclass
    class DataDataclass:
        a: int
        b: int
        c: int

try:
    import pydantic
except ImportError:
    print("pydantic tests disabled")
    pydantic = None
else:
    class DataPydanticModel(pydantic.BaseModel):
        a: Optional[Union[int, str]]
        b: Optional[Union[int, str]]
        c: Optional[Union[int, str]]

    class DataPydanticImmutableModel(pydantic.BaseModel):
        class Config:
            allow_mutation = False

        a: Optional[Union[int, str]]
        b: Optional[Union[int, str]]
        c: Optional[Union[int, str]]

    class DataPydanticORMModel(pydantic.BaseModel):
        class Config:
            orm_mode = True

        a: Optional[Union[int, str]]
        b: Optional[Union[int, str]]
        c: Optional[Union[int, str]]

try:
    import attr
except ImportError:
    print("attrs tests disabled")
    attr = None
else:
    AttrClass = attr.make_class("AttrClass", ["a", "b", "c"])

try:
    import traitlets
except ImportError:
    print("traitlets tests disabled")
    traitlets = None
else:
    class TraitletsClass(lt.HasTraitsMixin, traitlets.HasTraits):
        a = traitlets.Union([traitlets.Int(), traitlets.Unicode()], allow_none=True)
        b = traitlets.Union([traitlets.Int(), traitlets.Unicode()], allow_none=True)
        c = traitlets.Union([traitlets.Int(), traitlets.Unicode()], allow_none=True)

        def __init__(self, **kwargs):
            super().__init__()
            for k, w in kwargs.items():
                setattr(self, k, w)

        def __repr__(self):
            return "{}:(a={}, b={}, c={})".format(type(self).__name__, self.a, self.b, self.c)

        def __dir__(self):
            return self.trait_names()

DataTuple = namedtuple("DataTuple", "a b c")


class DataNamedTuple(NamedTuple):
    a: int
    b: int
    c: int


# if rich is not installed, disable table.present() calls
try:
    import rich
except ImportError:
    rich = None
    # disable present() method, since rich is not available
    lt.Table.present = lambda *args, **kwargs: None


class Slotted:
    __slots__ = ['a', 'b', 'c']

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __eq__(self, other):
        return (isinstance(other, Slotted) and
                all(getattr(self, attr) == getattr(other, attr) for attr in self.__slots__))

    def __repr__(self):
        return "{}:(a={}, b={}, c={})".format(type(self).__name__, self.a, self.b, self.c)


if python_version >= (3, 8):
    class SlottedWithDict:
        __slots__ = {'a': 'a', 'b': 'b', 'c': 'c'}

        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

        def __eq__(self, other):
            return (isinstance(other, SlottedWithDict) and
                    all(getattr(self, attr) == getattr(other, attr) for attr in self.__slots__))

        def __repr__(self):
            return "{}:(a={}, b={}, c={})".format(type(self).__name__, self.a, self.b, self.c)
else:
    SlottedWithDict = None


class TestDataObjects(unittest.TestCase):
    def test_set_attributes(self):
        ob = lt.DataObject()
        ob.z = 200
        ob.a = 100
        with self.subTest("test DataObject attribute setting"):
            self.assertEqual([('a', 100), ('z', 200)], sorted(ob.__dict__.items()))

        # test semi-immutability (can't overwrite existing attributes)
        with self.subTest("test DataObject write-once (semi-immutability)"):
            with self.assertRaises(AttributeError):
                ob.a = 101

        # equality tests
        with self.subTest("test DataObject equality"):
            ob2 = lt.DataObject(**{'a': 100, 'z': 200})
            self.assertEqual(ob2, ob)

        with self.subTest("test DataObject inequality"):
            ob2.b = 'blah'
            self.assertNotEqual(ob, ob2)

        with self.subTest("test DataObject equality after updates"):
            del ob2.b
            self.assertEqual(ob2, ob)

        with self.subTest("test DataObject KeyError"):
            del ob2.a
            del ob2.z

            with self.assertRaises(KeyError):
                ob2['a']
            ob2['a'] = 10
            ob2['a']

        with self.subTest("test DataObject KeyError (2)"):
            with self.assertRaises(KeyError):
                ob2['a'] = 10

        with self.subTest("test DataObject repr"):
            self.assertEqual("{'a': 10}", repr(ob2))


class TestTableTypes(unittest.TestCase):
    def test_types(self):

        # check that Table and Index are recognized as Sequence and Mapping types
        with self.subTest("check that Table is recognized as Sequence type"):
            t = lt.Table()
            self.assertTrue(isinstance(t, lt.Sequence))

        with self.subTest("check that Index is recognized as Mapping type"):
            t.create_index("x")
            self.assertTrue(isinstance(t.get_index('x'), lt.Mapping))

        # make sure get_index returns a read-only access to the underlying index
        with self.subTest("check that get_index returns a read-only access to the underlying index"):
            with self.assertRaises(lt.ReadonlyIndexAccessError):
                t.get_index("x")['a'] = 100


def announce_test(fn):
    def _inner(*args):
        print("\n" + "-" * 50)
        print("{}.{}".format(type(args[0]).__name__, fn.__name__))
        return fn(*args)

    return _inner


def make_test_class(*classes):

    class_name = "_".join(c.__name__ for c in classes)
    if not issubclass(classes[0], unittest.TestCase):
        cls = type(class_name, (unittest.TestCase, *classes), {})
    else:
        cls = type(class_name, tuple(classes), {})
    for attr in dir(cls):
        attrvalue = getattr(cls, attr)
        if attr.startswith("test_") and callable(attrvalue):
            setattr(cls, attr, announce_test(attrvalue))
    globals()[cls.__name__] = cls


def make_test_classes(cls):
    make_test_class(cls, UsingDataObjects)
    make_test_class(cls, UsingNamedtuples)
    make_test_class(cls, UsingDataNamedtuples)
    make_test_class(cls, UsingSlottedObjects)
    if SlottedWithDict is not None:
        make_test_class(cls, UsingSlottedWithDictObjects)
    make_test_class(cls, UsingSimpleNamespace)
    if dataclasses is not None:
        make_test_class(cls, UsingDataclasses)
    if pydantic is not None:
        make_test_class(cls, UsingPydanticModel)
        make_test_class(cls, UsingPydanticImmutableModel)
        make_test_class(cls, UsingPydanticORMModel)
    if attr is not None:
        make_test_class(cls, UsingAttrClass)
    if traitlets is not None:
        make_test_class(cls, UsingTraitletsClass)


class AbstractContentTypeFactory:
    data_object_type: Optional[type] = None

    @classmethod
    def make_data_object(cls, a, b, c):
        return cls.data_object_type(a=a, b=b, c=c)


class UsingDataObjects(AbstractContentTypeFactory):
    data_object_type = lt.DataObject


class UsingNamedtuples(AbstractContentTypeFactory):
    data_object_type = DataTuple


class UsingDataNamedtuples(AbstractContentTypeFactory):
    data_object_type = DataNamedTuple


class UsingSlottedObjects(AbstractContentTypeFactory):
    data_object_type = Slotted


if SlottedWithDict is not None:
    class UsingSlottedWithDictObjects(AbstractContentTypeFactory):
        data_object_type = SlottedWithDict
else:
    UsingSlottedWithDictObjects = AbstractContentTypeFactory


class UsingSimpleNamespace(AbstractContentTypeFactory):
    data_object_type = SimpleNamespace


if dataclasses is not None:
    class UsingDataclasses(AbstractContentTypeFactory):
        data_object_type = DataDataclass
else:
    UsingDataclasses = AbstractContentTypeFactory


if pydantic is not None:
    class UsingPydanticModel(AbstractContentTypeFactory):
        data_object_type = DataPydanticModel

    class UsingPydanticImmutableModel(AbstractContentTypeFactory):
        data_object_type = DataPydanticImmutableModel

    class UsingPydanticORMModel(AbstractContentTypeFactory):
        data_object_type = DataPydanticORMModel

else:
    UsingPydanticModel = AbstractContentTypeFactory
    UsingPydanticImmutableModel = AbstractContentTypeFactory
    UsingPydanticORMModel = AbstractContentTypeFactory

if attr is not None:
    class UsingAttrClass(AbstractContentTypeFactory):
        data_object_type = AttrClass
else:
    UsingAttrClass = AbstractContentTypeFactory

if traitlets is not None:
    class UsingTraitletsClass(AbstractContentTypeFactory):
        data_object_type = TraitletsClass
else:
    UsingTraitletsClass = AbstractContentTypeFactory


def load_table(table, rec_factory_fn, table_size):
    test_size = table_size
    for aa, bb, cc in itertools.product(range(test_size), repeat=3):
        table.insert(rec_factory_fn(aa, bb, cc))


def make_test_table(rec_factory_fn, table_size):
    table = lt.Table()
    load_table(table, rec_factory_fn, table_size)
    return table


def make_dataobject_from_ob(rec):
    return lt.DataObject(**dict((k, getattr(rec, k)) for k in lt._object_attrnames(rec)))


class TableTypeTests(unittest.TestCase):
    def test_types(self):
        from collections.abc import (Callable, Container, Iterable, Collection, Mapping, Reversible, Sequence, Sized)

        tbl = lt.Table()
        tbl.create_index("idx")

        for superclass in (Callable, Sized, Iterable, Container, Collection, Reversible, Sequence):
            with self.subTest(superclass=superclass):
                print(superclass.__name__)
                self.assertTrue(isinstance(tbl, superclass))

        with self.subTest():
            print("isinstance(_ObjIndex, Mapping)")
            self.assertTrue(isinstance(tbl._indexes["idx"], Mapping))

        with self.subTest():
            print("isinstance(_ObjIndexWrapper, Mapping)")
            self.assertTrue(isinstance(tbl.by.idx, Mapping))


@make_test_classes
class TableCreateTests:
    def test_inserts(self):
        table = lt.Table()
        table.insert(self.make_data_object(1, 2, 3))
        table.insert(self.make_data_object(4, 5, 6))
        table.create_index('a', unique=True)
        self.assertEqual(self.make_data_object(4, 5, 6), table.by.a[4])

        with self.assertRaises(KeyError):
            table.insert(self.make_data_object(4, 1, 0))

        with self.assertRaises(KeyError):
            table.insert(self.make_data_object(None, 1, 0))

        table.delete_index('a')
        table.insert(self.make_data_object(4, 1, 0))

        with self.assertRaises(KeyError):
            table.create_index('a', unique=True)

    def test_insert_dicts(self):
        table = lt.Table()
        table.insert({"a": 1, "b": 2, "c": 3})
        table.insert({"a": 4, "b": 5, "c": 6})
        table.create_index('a', unique=True)
        rec0, rec1 = table
        self.assertEqual({"a": 1, "b": 2, "c": 3}, vars(rec0))
        self.assertEqual(lt.default_row_class, type(rec0))
        self.assertEqual(1, getattr(rec0, "a"))

        # insert a nested dict
        table.clear()
        table.insert({"a": 1, "b": 2, "c": 3, "d": {"x": 100, "y": 200}})
        table.insert({"a": 4, "b": 5, "c": 6, "d": {"x": 101, "y": 201}})
        rec0, rec1 = table
        self.assertEqual(100, rec0.d.x)
        self.assertEqual(101, rec1.d.x)

    def test_where_equals(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)

        self.assertEqual(test_size*test_size, len(table.where(a=5)))
        self.assertEqual(0, len(table.where(a=-1)))

    def test_where_equals_none(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)

        self.assertEqual(0, len(table.where(a=5, b=test_size)))

    def test_where_equals_with_index(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)
        table.create_index('a')

        self.assertEqual(test_size*test_size, len(table.where(a=5)))
        self.assertEqual(0, len(table.where(a=-1)))

    def test_where_range(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)

        self.assertEqual(test_size*test_size, len(table.where(lambda rec: rec.a == rec.b)))

    def test_where_comparator(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)

        self.assertEqual(test_size*test_size*4, len(table.where(a=lt.Table.lt(4))))
        self.assertEqual(test_size*test_size*(4+1), len(table.where(a=lt.Table.le(4))))
        self.assertEqual(test_size*test_size*(test_size-4-1), len(table.where(a=lt.Table.gt(4))))
        self.assertEqual(test_size*test_size*(test_size-4), len(table.where(a=lt.Table.ge(4))))
        self.assertEqual(test_size*test_size*(test_size-1), len(table.where(a=lt.Table.ne(4))))
        self.assertEqual(test_size*test_size, len(table.where(a=lt.Table.eq(4))))
        self.assertEqual(test_size, len(table.where(a=lt.Table.eq(4), b=lt.Table.eq(4))))
        self.assertEqual(test_size*test_size*4, len(table.where(a=lt.Table.between(3, 8))))
        self.assertEqual(test_size*test_size*4, len(table.where(a=lt.Table.within(2, 5))))
        self.assertEqual(test_size*test_size*3, len(table.where(a=lt.Table.in_range(2, 5))))
        self.assertEqual(0, len(table.where(a=lt.Table.between(3, 3))))
        self.assertEqual(test_size*test_size, len(table.where(a=lt.Table.within(3, 3))))
        self.assertEqual(0, len(table.where(a=lt.Table.in_range(3, 3))))
        self.assertEqual(test_size*test_size*4, len(table.where(a=lt.Table.is_in([2, 4, 6, 8]))))
        self.assertEqual(0, len(table.where(a=lt.Table.is_in([]))))
        self.assertEqual(test_size*test_size*(test_size-4), len(table.where(a=lt.Table.not_in([2, 4, 6, 8]))))
        self.assertEqual(test_size*test_size*test_size, len(table.where(a=lt.Table.not_in([]))))

        # add a record containing a None value to test is_none and is_not_none comparators
        table.insert(self.make_data_object(a=1, b=2, c=None))
        self.assertEqual(1, len(table.where(c=lt.Table.is_none())))
        self.assertEqual(test_size*test_size*test_size, len(table.where(c=lt.Table.is_not_none())))

        # add a record containing a missing value to test is_null and is_not_null comparators
        table.insert(self.make_data_object(a=1, b=2, c=""))
        self.assertEqual(2, len(table.where(c=lt.Table.is_null())))
        self.assertEqual(test_size * test_size * test_size, len(table.where(c=lt.Table.is_not_null())))

        try:
            table.insert(self.make_data_object(a=1, b=2))
        except TypeError:
            # not all data object types being tested support missing attributes
            pass
        else:
            self.assertEqual(3, len(table.where(c=lt.Table.is_null())))
            self.assertEqual(test_size * test_size * test_size, len(table.where(c=lt.Table.is_not_null())))

    def test_where_str_comparator(self):
        unicode_numbers = lt.Table().csv_import(textwrap.dedent("""\
            name,code_value,numeric_value
            ROMAN NUMERAL ONE,8544,1
            ROMAN NUMERAL TWO,8545,2
            ROMAN NUMERAL THREE,8546,3
            ROMAN NUMERAL FOUR,8547,4
            ROMAN NUMERAL FIVE,8548,5
            ROMAN NUMERAL SIX,8549,6
            ROMAN NUMERAL SEVEN,8550,7
            ROMAN NUMERAL EIGHT,8551,8
            ROMAN NUMERAL NINE,8552,9
            ROMAN NUMERAL TEN,8553,10
            SUPERSCRIPT TWO,178,2
            SUPERSCRIPT THREE,179,3
            SUPERSCRIPT ONE,185,1
            SUPERSCRIPT ZERO,8304,0
            SUPERSCRIPT FOUR,8308,4
            SUPERSCRIPT FIVE,8309,5
            SUPERSCRIPT SIX,8310,6
            SUPERSCRIPT SEVEN,8311,7
            SUPERSCRIPT EIGHT,8312,8
            SUPERSCRIPT NINE,8313,9
            CIRCLED DIGIT ONE,9312,1
            CIRCLED DIGIT TWO,9313,2
            CIRCLED DIGIT THREE,9314,3
            CIRCLED DIGIT FOUR,9315,4
            CIRCLED DIGIT FIVE,9316,5
            CIRCLED DIGIT SIX,9317,6
            CIRCLED DIGIT SEVEN,9318,7
            CIRCLED DIGIT EIGHT,9319,8
            CIRCLED DIGIT NINE,9320,9
            CIRCLED DIGIT ZERO,9450,0
            """))

        ones = unicode_numbers.where(name=lt.Table.endswith("ONE"))
        self.assertEqual(3, len(ones))

        supers = unicode_numbers.where(name=lt.Table.startswith("SUPERSCRIPT"))
        self.assertEqual(10, len(supers))

        # import re
        # sevens = unicode_numbers.where(lambda rec: re.compile(r".*SEVEN$").match(rec.name))

        sevens = unicode_numbers.where(name=lt.Table.re_match(r".*SEVEN$"))
        self.assertEqual(3, len(sevens))

        # make names all title case
        unicode_numbers.add_field("name", lambda rec: rec.name.title())
        # use regex with re flag
        import re
        circled = unicode_numbers.where(name=lt.Table.re_match(r"circled", flags=re.I))
        self.assertEqual(10, len(circled))

    def test_where_attr_function(self):
        test_size = 8
        table = make_test_table(self.make_data_object, test_size)

        def is_odd(x):
            return bool(x % 2)

        self.assertEqual(test_size*test_size*test_size//2, len(table.where(a=is_odd)))

    def test_get_slice(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)

        subtable = table[0::test_size]
        self.assertEqual(test_size * test_size, len(subtable))

    def test_indexing(self):
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        make_rec = lambda aa, bb, cc: self.make_data_object(chars[aa], chars[bb], chars[cc])
        test_size = 10
        table = make_test_table(make_rec, test_size)
        table.create_index('a')

        self.assertTrue('A' in table.by.a)
        self.assertTrue('AA' not in table.by.a)
        self.assertEqual(test_size * test_size, len(table.by.a['B']))
        self.assertTrue(isinstance(table.by.a['B'], lt.Table))
        with self.assertRaises(AttributeError):
            table.by.z

        self.assertEqual(test_size, len(table.by.a.keys()))

    def test_unique_indexing(self):
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        make_unique_key = lambda *args: ''.join(chars[arg] for arg in args)
        make_rec = lambda aa, bb, cc: self.make_data_object(make_unique_key(aa, bb, cc), chars[bb], chars[cc])
        test_size = 10
        table = make_test_table(make_rec, test_size)("Table_1")
        table.create_index('a', unique=True)
        rec_type = type(self.make_data_object(0, 0, 0))

        self.assertTrue('AAA' in table.by.a)
        self.assertTrue('AA' not in table.by.a)
        self.assertTrue(isinstance(table.by.a['BAA'], rec_type))
        with self.assertRaises(KeyError):
            table.insert(self.make_data_object(None, None, None))

        # create duplicate index
        with self.assertRaises(ValueError):
            table.create_index('a', unique=True, accept_none=True)

        # create unique index that allows None values
        table.delete_index('a')
        table.create_index('a', unique=True, accept_none=True)
        table.insert(self.make_data_object(None, None, 'A'))

        str_none_compare = lambda x: x if isinstance(x, str) else chr(255)*100
        self.assertEqual(sorted(table.by.a.keys(), key=str_none_compare),
                         sorted(table.all.a, key=str_none_compare))

        # now drop index and recreate not permitting None, should raise exception
        table.delete_index('a')
        with self.assertRaises(KeyError):
            table.create_index('a', unique=True, accept_none=False)

        table.create_index('a', unique=True, accept_none=True)
        table.create_index('c')

        import pprint
        info = table.info()
        pprint.pprint(info)
        self.assertEqual('Table_1', info['name'])
        self.assertEqual(['a', 'b', 'c'], list(sorted(info['fields'])))
        self.assertEqual([('a', True), ('c', False)], list(sorted(info['indexes'])))
        self.assertEqual(1001, info['len'])

    def test_chained_indexing(self):
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        make_rec = lambda aa, bb, cc: self.make_data_object(chars[aa % len(chars)],
                                                            chars[bb % len(chars)],
                                                            chars[cc % len(chars)])
        test_size = 10
        table = make_test_table(make_rec, test_size)
        table.create_index('a')
        table.create_index('b')
        table.create_index('c')

        chained_table = table.by.b['A'].by.c['C']
        for rec in chained_table:
            print(rec)

        self.assertEqual(test_size, len(chained_table))

    def test_parse_datetime_transform(self):
        import datetime

        data = textwrap.dedent("""\
        a,b,c
        2001-01-01 00:34:56,A,100
        2001-01-02 01:34:56,B,101
        2001-02-30 02:34:56,C,102
        ,D,103
        """)
        test_kwargs = [
            {'empty': '', 'on_error': None},
            {'empty': 'N/A', 'on_error': datetime.datetime.min},
            {'empty': datetime.datetime.min, 'on_error': ''},
        ]
        for kwargs in test_kwargs:
            tbl = lt.Table().csv_import(data,
                                        transforms={'a': lt.Table.parse_datetime('%Y-%m-%d %H:%M:%S',
                                                                                 **kwargs)})
            print([str(a) for a in tbl.all.a])

            with self.subTest("test Table.parse_date_time errors", **kwargs):
                self.assertEqual(
                    [kwargs["on_error"], kwargs["empty"]],
                    list(tbl.all.a)[-2:]
                )

            with self.subTest("test Table.parse_date_time valid", **kwargs):
                self.assertEqual(
                    [datetime.datetime(2001, 1, 1, 0, 34, 56),
                     datetime.datetime(2001, 1, 2, 1, 34, 56)],
                    list(tbl.all.a)[:2]
                )

    def test_parse_date_transform(self):
        import datetime

        data = textwrap.dedent("""\
        a,b,c
        2001-01-01 00:34:56,A,100
        2001-01-02 01:34:56,B,101
        2001-02-30 02:34:56,C,102
        ,D,103
        """)
        test_kwargs = [
            {'empty': '', 'on_error': None},
            {'empty': 'N/A', 'on_error': datetime.date.min},
            {'empty': datetime.date.min, 'on_error': ''},
        ]
        for kwargs in test_kwargs:
            tbl = lt.Table().csv_import(data,
                                        transforms={'a': lt.Table.parse_date('%Y-%m-%d %H:%M:%S',
                                                                             **kwargs)})
            print([str(a) for a in tbl.all.a])

            with self.subTest("test Table.parse_date_time errors", **kwargs):
                self.assertEqual(
                    [kwargs["on_error"], kwargs["empty"]],
                    list(tbl.all.a)[-2:]
                )

            with self.subTest("test Table.parse_date_time valid", **kwargs):
                self.assertEqual(
                    [datetime.date(2001, 1, 1),
                     datetime.date(2001, 1, 2)],
                    list(tbl.all.a)[:2]
                )

    def test_parse_timedelta_transform(self):
        import datetime

        process_data = textwrap.dedent("""\
            elapsed_time,eqpt,event,lot,pieces
            0:00:00,DRILL01,LotStart,PCB146,1
            0:00:40,DRILL01,Tool1,PCB146,2
            0:03:45,DRILL01,Tool2,PCB146,4
            0:06:16,DRILL01,LotEnd,PCB146,8
            """)

        transforms = {'elapsed_time': lt.Table.parse_timedelta("%H:%M:%S"),
                      'pieces': int}
        data = lt.Table(f"Process step elapsed times").csv_import(process_data, transforms=transforms)
        data.create_index("elapsed_time")

        _00_01_30 = datetime.timedelta(seconds=90)
        self.assertEqual(3, sum(data.by.elapsed_time[:_00_01_30].all.pieces))

    def test_sliced_indexing(self):
        transforms = {
            'pop': int,
            'elev': int,
            'lat': float,
            'long': float,
        }
        us_ppl = lt.Table().csv_import("examples/us_ppl.csv",
                                       transforms=transforms,
                                       ).select("id name elev lat long pop")
        print(us_ppl.info())
        us_ppl.create_index("name")
        us_ppl.create_index("elev")

        test = "elev < 0"
        low_ppl_where = us_ppl.where(elev=lt.Table.lt(0))(test)
        low_ppl_slice = us_ppl.by.elev[:0](f"{test} (sliced)")
        low_ppl_slice.present()
        self.assertEqual(list(low_ppl_where.all.id), list(low_ppl_slice.all.id))

        test = "elev >= 1000"
        hi_ppl_where = us_ppl.where(elev=lt.Table.ge(1000))(test)
        hi_ppl_slice = us_ppl.by.elev[1000:](f"{test} (sliced)")
        self.assertEqual(list(hi_ppl_where.all.id), list(hi_ppl_slice.all.id))

        test = "0 <= elev < 100"
        low_ppl_where = us_ppl.where(elev=lt.Table.ge(0)).where(elev=lt.Table.lt(100))(test)
        low_ppl_slice = us_ppl.by.elev[0:100](f"{test} (sliced)")
        self.assertEqual(list(low_ppl_where.all.id), list(low_ppl_slice.all.id))

        a_ppl_where = us_ppl.where(name=lt.Table.ge("A")).where(name=lt.Table.lt("C"))
        a_ppl_slice = us_ppl.by.name["A":"C"]
        self.assertEqual(list(a_ppl_where.all.id), list(a_ppl_slice.all.id))

    def test_non_integer_sliced_indexing(self):
        import datetime

        sales_data = textwrap.dedent("""\
            date,customer,sku,qty
            2000/01/01,0020,ANVIL-001,1
            2000/01/01,0020,BRDSD-001,2
            2000/02/15,0020,BRDSD-001,4
            2000/03/31,0020,BRDSD-001,8
            2000/03/31,0020,MAGNT-001,16
            2000/04/01,0020,ROBOT-001,32
            2000/04/15,0020,BRDSD-001,64
            """)

        transforms = {'date': lt.Table.parse_date("%Y/%m/%d"),
                      'qty': int}
        sales = lt.Table().csv_import(sales_data,
                                      transforms=transforms,)

        sales.create_index("date")
        jan_01 = datetime.date(2000, 1, 1)
        apr_01 = datetime.date(2000, 4, 1)
        first_qtr_sales = sales.by.date[jan_01: apr_01]
        first_qtr_sales.present()
        print(list(first_qtr_sales.all.sku))

        self.assertEqual(list(first_qtr_sales.all.sku),
                         ['ANVIL-001', 'BRDSD-001', 'BRDSD-001', 'BRDSD-001', 'MAGNT-001'],
                         )
        self.assertEqual(31, sum(first_qtr_sales.all.qty))

        # use date strings as range values
        transforms = {'qty': int}
        sales = lt.Table().csv_import(sales_data,
                                      transforms=transforms,)

        sales.create_index("date")
        first_qtr_sales = sales.by.date["2000/01/01": "2000/04/01"]
        first_qtr_sales.present()
        print(list(first_qtr_sales.all.sku))

        self.assertEqual(list(first_qtr_sales.all.sku),
                         ['ANVIL-001', 'BRDSD-001', 'BRDSD-001', 'BRDSD-001', 'MAGNT-001'],
                         )
        self.assertEqual(31, sum(first_qtr_sales.all.qty))

        self.assertEqual(31, sum(sales.by.date[:"2000/04/01"].all.qty))
        self.assertEqual(96, sum(sales.by.date["2000/04/01":].all.qty))

    def test_index_dir(self):
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        make_rec = lambda aa, bb, cc: self.make_data_object(chars[aa % len(chars)],
                                                            chars[bb % len(chars)],
                                                            chars[cc % len(chars)])
        test_size = 10
        table = make_test_table(make_rec, test_size)
        table.create_index('a')
        table.create_index('b')

        dir_list = dir(table.by)
        print([attr for attr in dir_list if not attr.startswith("_")])

        self.assertTrue(('a' in dir_list) and ('b' in dir_list) and ('c' not in dir_list))

    def test_delete_by_filter(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)

        self.assertEqual(test_size*test_size, table.delete(b=5))
        self.assertEqual(test_size*test_size*(test_size-1), len(table))
        self.assertEqual(0, table.delete(b=-1))
        self.assertEqual(0, table.delete())

    def test_remove_many(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)

        self.assertEqual(test_size*test_size*test_size/2, len(table.where(lambda t: t.a % 2)))
        table.remove_many(table.where(lambda t: t.a % 2))
        self.assertEqual(test_size*test_size*test_size/2, len(table))
        table_len = len(table)
        table.remove(table[1])
        self.assertEqual(table_len-1, len(table))

    def test_add_new_field(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)

        # only DataObjects are mutable in these tests
        if isinstance(table[0], lt.DataObject):
            table.add_field('d', lambda rec: rec.a+rec.b+rec.c)

            table.create_index('d')
            self.assertEqual(len(range(0, 27+1)), len(table.by.d.keys()))

    def test_add_two_tables(self):
        test_size = 10
        t1 = make_test_table(self.make_data_object, test_size)
        make_rec = lambda a,b,c: self.make_data_object(a+test_size, b, c)
        t2 = make_test_table(make_rec, test_size)

        self.assertEqual(test_size*test_size*test_size*2, len(t1+t2))
        self.assertEqual(test_size * test_size * test_size, len(t1))

        t1 += t2
        self.assertEqual(test_size * test_size * test_size * 2, len(t1))

        offset = test_size * test_size
        t3 = t1 + (self.make_data_object(rec.a+offset, rec.b, rec.c) for rec in t2)
        self.assertEqual(test_size * test_size * test_size * 3, len(t3))

    def test_table_info(self):
        test_size = 10
        t1 = make_test_table(self.make_data_object, test_size)('info_test')
        t1.create_index('b')
        t1_info = t1.info()
        # must sort fields and indexes values, for test comparisons
        t1_info['fields'].sort()
        t1_info['indexes'].sort()
        self.assertEqual({'fields': ['a', 'b', 'c'],
                          'indexes': [('b', False)],
                          'len': 1000,
                          'name': 'info_test'},
                         t1_info, "invalid info results")


@make_test_classes
class TableListTests:
    def _test_init(self):
        self.test_size = 3
        self.t1 = make_test_table(self.make_data_object, self.test_size)
        self.test_rec = self.make_data_object(1, 1, 1)

    def test_contains(self):
        self._test_init()
        self.assertTrue(self.test_rec in self.t1, "failed 'in' (contains) test")

    def test_index_find(self):
        self._test_init()
        self.assertEqual(13, self.t1.index(self.test_rec), "failed 'index; test (exists)")
        if isinstance(self.test_rec, SimpleNamespace):
            self.assertEqual(13, self.t1.index(vars(self.test_rec)), "failed 'index; test (exists)")

        no_such_rec = self.make_data_object(self.test_size+1, self.test_size+1, self.test_size+1)
        with self.assertRaises(ValueError, msg="failed 'index' test (not exists)"):
            self.t1.index(no_such_rec)

    def test_remove(self):
        self._test_init()
        rec = self.make_data_object(1, 1, 1)
        prev_len = len(self.t1)
        self.t1.remove(rec)
        self.assertFalse(rec in self.t1, "failed to remove record from table (contains)")
        self.assertEqual(prev_len-1, len(self.t1), "failed to remove record from table (len)")

        no_such_rec = self.make_data_object(self.test_size+1, self.test_size+1, self.test_size+1)
        self.assertFalse(no_such_rec in self.t1, "failed to create non-existent record from table")
        self.t1.remove(no_such_rec)
        self.assertEqual(prev_len-1, len(self.t1), "failed removing non-existent record from table (len)")

        if isinstance(self.test_rec, SimpleNamespace):
            self.t1.remove(vars(self.test_rec))
            self.assertEqual(prev_len-1, len(self.t1), "failed to remove record as dict from table (len)")

    def test_index_access(self):
        self._test_init()
        self.assertEqual(self.test_rec, self.t1[13], "failed index access test")

    def test_count(self):
        self._test_init()
        self.assertTrue(self.t1.count(self.test_rec) == 1, "failed count test")
        if isinstance(self.test_rec, SimpleNamespace):
            self.assertTrue(self.t1.count(vars(self.test_rec)) == 1, "failed count test")

    def test_del(self):
        self._test_init()
        before_del_len = len(self.t1)
        del self.t1[13]
        self.assertFalse(self.test_rec in self.t1, "failed del test")
        self.assertEqual(before_del_len - 1, len(self.t1))

    def test_pop(self):
        self._test_init()
        before_pop_len = len(self.t1)
        obj = self.t1.pop(13)
        self.assertFalse(obj in self.t1)
        self.assertEqual(self.test_rec, obj)
        self.assertEqual(before_pop_len - 1, len(self.t1))

    def test_pop_last(self):
        self._test_init()
        before_pop_len = len(self.t1)
        expected_pop = copy.copy(self.t1[-1])
        obj = self.t1.pop()
        self.assertEqual(expected_pop, obj)
        self.assertEqual(before_pop_len - 1, len(self.t1))

    def test_reversed(self):
        self._test_init()
        last_rec = next(reversed(self.t1))
        self.assertEqual(self.make_data_object(2, 2, 2), last_rec, "failed reversed test")

    def test_iter(self):
        self._test_init()
        self.assertTrue(self.test_rec in self.t1, "failed 'in' (contains) test")

    def test_head_and_tail(self):
        self._test_init()
        self.t1.create_index("a")
        self.t1.create_index("c")
        self.assertEqual({"a", "c"}, set(self.t1._indexes.keys()), "failed to create indexes")

        self.assertEqual(set(self.t1._indexes.keys()),
                         set(self.t1.head()._indexes.keys()),
                         "failed to copy indexes to head()")
        self.assertEqual(set(self.t1._indexes.keys()),
                         set(self.t1.tail()._indexes.keys()),
                         "failed to copy indexes to tail()")

    def test_unique(self):
        self._test_init()
        self.assertEqual([0, 1, 2], sorted([row.a for row in self.t1.unique('a')]), "failed call to unique")

    def test_all_accessor(self):
        self._test_init()
        self.assertEqual(sum(([i]*self.test_size**2 for i in range(self.test_size)), []),
                         list(self.t1.all.a),
                         "failed to successfully get all values in 'a'")

        all_as = self.t1.all.a
        self.assertTrue(all_as is iter(all_as), "all iterator fails to identify as iter(self)")

    def test_format(self):
        self._test_init()
        self.assertEqual(['00 0 0', '00 0 1', '00 0 2'],
                         list(self.t1.format("{a:02d} {b} {c}"))[:3],
                         "failed to create formatted rows")

    def test_as_html(self):
        self._test_init()
        html_output = self.t1[:10].as_html(fields="a b c")
        print(html_output)
        self.assertTrue("<thead>" in html_output and "<tbody>" in html_output,
                        "as_html does not include thead and tbody tags")

        html_lines = html_output.splitlines()
        hdr_line = next(h for h in html_lines if "center" in h)
        self.assertEqual('<tr><th><div align="center">a</div></th>'
                         '<th><div align="center">b</div></th>'
                         '<th><div align="center">c</div></th></tr>',
                         hdr_line,
                         "failed as_html with all fields")

        html_output = self.t1[:10].as_html(fields="a -b c")
        print(html_output)
        html_lines = html_output.splitlines()
        hdr_line = next(h for h in html_lines if "center" in h)
        self.assertEqual('<tr><th><div align="center">a</div></th>'
                         '<th><div align="center">c</div></th></tr>',
                         hdr_line,
                         "failed as_html with negated field")

        html_output = self.t1[:10].as_html(fields="a b c", formats={"b": "{:03d}"})
        print(html_output)
        html_lines = html_output.splitlines()
        data_line = next(h for h in html_lines if "<td>" in h)
        self.assertEqual('<tbody><tr><td><div align="right">0</div></td>'
                         '<td><div align="right">000</div></td>'
                         '<td><div align="right">0</div></td></tr>',
                         data_line,
                         "failed as_html with named field format for specific field")

        html_output = self.t1[:10].as_html(fields="a b c", formats={int: "{:03d}"})
        print(html_output)
        html_lines = html_output.splitlines()
        data_line = next(h for h in html_lines if "<td>" in h)
        self.assertEqual('<tbody><tr><td><div align="right">000</div></td>'
                         '<td><div align="right">000</div></td>'
                         '<td><div align="right">000</div></td></tr>',
                         data_line,
                         "failed as_html with data type format for all fields")

        html_output = self.t1[:10].as_html(fields="a b c", formats={int: "{:03d}"}, groupby="a")
        print(html_output)
        html_lines = html_output.splitlines()
        data_line = next(h for h in html_lines if "<td>" in h)
        self.assertEqual('<tbody><tr><td><div align="right">000</div></td>'
                         '<td><div align="right">000</div></td>'
                         '<td><div align="right">000</div></td></tr>',
                         data_line,
                         "failed as_html with groupby")

        html_output = self.t1[:10].as_html(fields="a b c", table_properties={"border": "2"})
        print(html_output)
        self.assertTrue("<thead>" in html_output and "<tbody>" in html_output,
                        "as_html does not include thead and tbody tags")


    def test_delete_slices(self):
        compare_list = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz")
        t1 = lt.Table().insert_many(lt.DataObject(A=c) for c in compare_list)

        def mini_test(slc_tuple):
            if isinstance(slc_tuple, tuple):
                label = "[{}:{}:{}]".format(*(i if i is not None else '' for i in slc_tuple))
                slc = slice(*slc_tuple)
            else:
                label = str(slc_tuple)
                slc = slc_tuple

            with self.subTest(label, slc_tuple=slc_tuple):
                del compare_list[slc]
                print(label)
                print('Expected', compare_list)
                del t1[slc]
                print('Observed', list(t1.all.A))
                print()
                self.assertEqual(''.join(compare_list), ''.join(t1.all.A), "failed " + label)

        mini_test(5)
        mini_test(-5)
        mini_test((-5, None, None))
        mini_test((None, 3, None))
        mini_test((None, -len(compare_list)+3, None))
        mini_test((None, None, 5))
        mini_test((None, None, -5))
        mini_test((-5, -2, None))
        mini_test((-2, -5, -1))
        mini_test((5, 20, 2))
        mini_test((len(compare_list), 5, -3))
        mini_test((20, 11, -7))
        mini_test((5, 5, None))
        mini_test((None, -10, 5))
        mini_test((None, -10, -10))
        mini_test((1000, 2000, None))
        mini_test((None, None, -1))

    def test_clear(self):
        self._test_init()
        num_fields = len(self.t1.info()["fields"])
        with self.subTest():
            self.assertEqual(self.test_size ** num_fields, len(self.t1), "invalid len")
        self.t1.create_index("a")

        self.t1.clear()
        with self.subTest():
            self.assertEqual(0, len(self.t1), "invalid len after clear")
        with self.subTest():
            self.assertEqual(1, len(self.t1.info()["indexes"]), "invalid indexes after clear")

    def test_stats(self):
        self._test_init()
        field_names = self.t1.info()["fields"]
        num_fields = len(field_names)
        t1_stats = self.t1.stats().select("name count min max mean")
        for fieldname in field_names:
            stat_rec = t1_stats.by.name[fieldname]
            with self.subTest("check computed stat", fieldname=fieldname):
                self.assertEqual(lt.DataObject(name=fieldname,
                                               count=self.test_size ** num_fields,
                                               min=0,
                                               max=self.test_size - 1,
                                               mean=(self.test_size - 1) / 2),
                                 stat_rec,
                                 "invalid stat for {}".format(fieldname))

    def test_stats2(self):
        self._test_init()
        field_names = self.t1.info()["fields"]
        num_fields = len(field_names)
        t1_stats = self.t1.stats(by_field=False)
        for stat, value in (('min', 0), ('max', self.test_size - 1), ('count', self.test_size ** num_fields),):
            for fieldname in field_names:
                with self.subTest("check computed stat", stat=stat, fieldname=fieldname):
                    self.assertEqual(value, getattr(t1_stats.by.stat[stat], fieldname),
                                 "invalid {} stat for {}".format(stat, fieldname))

    def test_stats3(self):
        self._test_init()
        field_names = self.t1.info()["fields"]
        num_fields = len(field_names)

        # verify that stats can "step over" non-numeric data
        try:
            self.t1[0].a = "not a number"
        except (AttributeError, TypeError):
            # some test types aren't mutable, must replace rec with a modified one
            mod_rec = self.t1.pop(0)
            rec_type = type(mod_rec)
            new_rec_dict = lt._to_dict(mod_rec)
            new_rec_dict['a'] = "not a number"
            new_rec = rec_type(**new_rec_dict)
            self.t1.insert(new_rec)

        t1_stats = self.t1.stats()
        t1_stats("t1_stats")
        t1_stats.csv_export(sys.stdout)
        t1_stats.present()
        self.assertEqual(self.test_size ** num_fields - 1, t1_stats.by.name["a"].count)

    def test_stats4(self):
        t1 = lt.Table().csv_import(textwrap.dedent("""\
        a,b
        1,2
        3,
        5,4
        """), transforms={}.fromkeys(["a", "b"], int))
        t1_stats = t1.stats()
        t1_stats.present()
        print(t1_stats.info())

        expected = lt.Table().csv_import(textwrap.dedent("""\
        name,mean,min,max,variance,std_dev,count,missing
        a,3.0,1,5,4,2.0,3,0
        b,3.0,2,4,2,1.414,2,1
        """), transforms={}.fromkeys("mean min max variance std_dev count missing".split(), ast.literal_eval))
        expected.present()
        print(expected.info())

        with self.subTest("check computed stat fields"):
            self.assertEqual(expected.info()["fields"], t1_stats.info()["fields"])

        for expected_row, row in zip(expected, t1_stats):
            with self.subTest("check computed stat attribute (name)", row=row):
                self.assertEqual(expected_row.name, row.name)
            with self.subTest("check computed stat attribute (mean)", row=row):
                self.assertEqual(expected_row.mean, row.mean)
            with self.subTest("check computed stat attribute (min)", row=row):
                self.assertEqual(expected_row.min, row.min)
            with self.subTest("check computed stat attribute (max)", row=row):
                self.assertEqual(expected_row.max, row.max)
            with self.subTest("check computed stat attribute (variance)", row=row):
                self.assertEqual(expected_row.variance, row.variance)
            with self.subTest("check computed stat attribute (std_dev)", row=row):
                self.assertEqual(expected_row.std_dev, row.std_dev)
            with self.subTest("check computed stat attribute (count)", row=row):
                self.assertEqual(expected_row.count, row.count)
            with self.subTest("check computed stat attribute (missing)", row=row):
                self.assertEqual(expected_row.missing, row.missing)

    def test_splitby(self):
        self._test_init()
        is_odd = lambda rec: rec.a % 2
        evens, odds = self.t1.splitby(is_odd)
        with self.subTest():
            self.assertEqual(len(odds) + len(evens), len(self.t1))
        with self.subTest():
            self.assertEqual(len(odds), len(self.t1.where(is_odd)))

        even_evens, odd_evens = evens.splitby(is_odd)
        with self.subTest():
            self.assertEqual(0, len(odd_evens))
        with self.subTest():
            self.assertEqual(len(even_evens), len(evens))

        # make sure indexes are preserved
        self.t1.create_index("a")
        evens, odds = self.t1.splitby(is_odd)
        with self.subTest():
            self.assertEqual(self.t1.info()["indexes"], evens.info()["indexes"])

        # test passing an attribute as a key
        zeros, non_zeros = self.t1.splitby("a")
        with self.subTest():
            self.assertTrue(all(rec.a == 0 for rec in zeros))
        with self.subTest():
            self.assertTrue(all(rec.a != 0 for rec in non_zeros))

        # test using predicate that does not always return 0 or 1
        is_not_multiple_of_3 = lambda rec: rec.a % 3
        mults_of_3, non_mults_of_3 = self.t1.splitby(is_not_multiple_of_3)
        print(list(non_mults_of_3.all.a))
        print(list(mults_of_3.all.a))
        # TODO - add asserts here


@make_test_classes
class TableJoinTests:
    def test_simple_join(self):
        test_size = 10
        t1 = make_test_table(self.make_data_object, test_size)
        t1.create_index('a')

        t2 = lt.Table()
        t2.create_index('a')
        t2.insert(lt.DataObject(a=1, d=100))

        joined = (t1.join_on('a') + t2.join_on('a'))()
        with self.subTest():
            self.assertEqual(test_size * test_size, len(joined))

        joined = (t1.join_on('a') + t2)()
        with self.subTest():
            self.assertEqual(test_size * test_size, len(joined))

        joined = (t1 + t2.join_on('a'))()
        with self.subTest():
            self.assertEqual(test_size * test_size, len(joined))

        t1.delete_index('a')
        with self.subTest():
            with self.assertRaises(ValueError):
                joined = (t1 + t2.join_on('a'))()

        with self.subTest():
            with self.assertRaises(TypeError):
                # invalid join, no kwargs listing attributes to join on
                t3 = t1.join(t2, 'a,d')

        with self.subTest():
            with self.assertRaises(ValueError):
                # invalid join, no such attribute 'z'
                t3 = t1.join(t2, 'a,d,z', a='a')

        t3 = t1.join(t2, 'a,d', a='a')
        with self.subTest():
            self.assertEqual(test_size * test_size, len(t3))

        t4 = t1.join(t2, a='a').select('a c d', e=lambda rec: rec.a + rec.c + rec.d)
        with self.subTest():
            self.assertTrue(all(rec.e == rec.a+rec.c+rec.d for rec in t4))

        # join to empty list, should return empty table
        empty_table = lt.Table()
        empty_table.create_index('a')
        t5 = (t1.join_on('a') + empty_table)()
        with self.subTest():
            self.assertEqual(0, len(t5))

    def test_outer_joins(self):
        t1 = lt.Table("catalog")
        t1.csv_import(textwrap.dedent("""\
            sku,color,size,material
            001,red,XL,cotton
            002,blue,XL,cotton/poly
            003,blue,L,linen
            004,red,M,cotton
            """))

        t2 = lt.Table("prices")
        t2.csv_import(textwrap.dedent("""\
            sku,unit_price,size
            001,10,L
            001,12,XL
            002,11,
            004,9,
            """), transforms={'size': lambda x: x or None})
        print(t1.info())

        t1.present()
        t2.present()

        t3 = t1.join(t2, auto_create_indexes=True, sku="sku")
        print(t3.info())
        t3.present()
        with self.subTest():
            self.assertEqual(4, len(t3))

        t3 = t1.join(t2, auto_create_indexes=True, sku="sku", size="size")
        t3("inner join - " + t3.table_name)
        print(t3.info())
        t3.present()
        with self.subTest():
            self.assertEqual(1, len(t3))

        t3 = t1.outer_join(lt.Table.RIGHT_OUTER_JOIN, t2, sku="sku", size="size")
        t3("right outer join - " + t3.table_name)
        print(t3.info())
        t3.present()
        with self.subTest():
            self.assertEqual(4, len(t3))

        t3 = t1.outer_join(lt.Table.LEFT_OUTER_JOIN, t2, sku="sku", size="size")
        t3("left outer join - " + t3.table_name)
        print(t3.info())
        t3.present()
        with self.subTest():
            self.assertEqual(2, len(t3))

        t3 = t1.outer_join(lt.Table.FULL_OUTER_JOIN, t2, sku="sku", size="size")
        t3("full outer join - " + t3.table_name)
        print(t3.info())
        t3.present()
        with self.subTest():
            self.assertEqual(12, len(t3))

    def test_outer_join_example(self):
        # define student and registration data
        students = lt.Table("students").csv_import(textwrap.dedent("""\
            student_id,name
            0001,Alice
            0002,Bob
            0003,Charlie
            0004,Dave
            0005,Enid
            """))

        registrations = lt.Table("registrations").csv_import(textwrap.dedent("""\
            student_id,course
            0001,PSYCH101
            0001,CALC1
            0003,BIO200
            0005,CHEM101
            0006,PHY101
            """))

        courses = lt.Table("courses").csv_import(textwrap.dedent("""\
            course
            BIO200
            CALC1
            CHEM101
            PSYCH101
            PE101
            """))

        # perform outer join and show results:
        non_reg = students.outer_join(lt.Table.RIGHT_OUTER_JOIN,
                                      registrations,
                                      student_id="student_id").where(course=None)
        non_reg.present()
        print(list(non_reg.all.name))
        with self.subTest():
            self.assertEqual(['Bob', 'Dave'], sorted(non_reg.all.name))

        # courses with no students
        no_students = registrations.outer_join(lt.Table.LEFT_OUTER_JOIN,
                                         courses,
                                         course="course").where(student_id=None)
        no_students.present()
        print(list(no_students.all.course))
        with self.subTest():
            self.assertEqual(['PE101'], sorted(no_students.all.course))


        full =  students.outer_join(lt.Table.FULL_OUTER_JOIN,
                                      registrations,
                                      student_id="student_id").where(lambda rec: rec.course is None
                                                                                 or rec.name is None)
        full.present()
        print(sorted(full.all.student_id))
        with self.subTest():
            self.assertEqual(['0002', '0004', '0006'], sorted(full.all.student_id))


@make_test_classes
class TableTransformTests:
    def test_sort(self):
        test_size = 10
        t1 = make_test_table(self.make_data_object, test_size)

        c_groups = 0
        for c_value, recs in itertools.groupby(t1, key=lambda rec: rec.c):
            c_groups += 1
            list(recs)
        with self.subTest():
            self.assertEqual(test_size * test_size * test_size, c_groups)

        t1.sort('c')
        c_groups = 0
        for c_value, recs in itertools.groupby(t1, key=lambda rec: rec.c):
            c_groups += 1
            list(recs)
        with self.subTest():
            self.assertEqual(test_size, c_groups)
        with self.subTest():
            self.assertEqual(0, t1[0].c)

        t1.sort('c desc')
        with self.subTest():
            self.assertEqual(test_size-1, t1[0].c)

    def test_sort2(self):

        row_type = type(self.make_data_object(0,0,0))
        tt = lt.Table().csv_import(textwrap.dedent("""\
        a,c,b
        1,2,1
        2,3,0
        5,5,-1
        3,4,-1
        2,4,-3"""), row_class=row_type, transforms=dict.fromkeys("a b c".split(), int))

        def to_tuples(t):
            return list(map(attrgetter(*t.info()['fields']), t))

        print(tt.info()['fields'])

        sort_arg = "c b".split()
        print("Sorting by {!r}".format(sort_arg))
        tt.shuffle()
        tt.sort(sort_arg)
        t1_tuples = to_tuples(tt)
        for t in t1_tuples:
            print(t)
        print()

        tt.shuffle()
        tt.sort("b")
        tt.sort("c")
        t2_tuples = to_tuples(tt)
        for t in t2_tuples:
            print(t)
        print()

        with self.subTest():
            self.assertEqual(t1_tuples, t2_tuples, "failed multi-attribute sort, given list of attributes")

        sort_arg = "c,b"
        print("Sorting by {!r}".format(sort_arg))
        tt.shuffle()
        tt.sort(sort_arg)
        t1_tuples = to_tuples(tt)
        for t in t1_tuples:
            print(t)
        print()

        tt.shuffle()
        tt.sort("b")
        tt.sort("c")
        t2_tuples = to_tuples(tt)
        for t in t2_tuples:
            print(t)
        print()

        with self.subTest():
            self.assertEqual(t1_tuples, t2_tuples, "failed multi-attribute sort, given comma-separated attributes string")

        sort_arg = "c,b desc"
        print("Sorting by {!r}".format(sort_arg))
        tt.shuffle()
        tt.sort(sort_arg)
        t1_tuples = to_tuples(tt)
        for t in t1_tuples:
            print(t)
        print()

        tt.shuffle()
        tt.sort("b desc")
        tt.sort("c")
        t2_tuples = to_tuples(tt)
        for t in t2_tuples:
            print(t)
        print()

        with self.subTest():
            self.assertEqual(t1_tuples, t2_tuples, "failed mixed ascending/descending multi-attribute sort")

    def test_sort3(self):
        employees = lt.Table().csv_import(textwrap.dedent("""\
            emp_id,name,dept,salary,commission
            0001,Alice,Sales,50000,0.5
            0002,Bob,Engineering,100000,
            0003,Charles,Sales,45000,0.7
            0004,Dave,Sales,45000,0.6
            0005,Emily,Sales,50000,0.4
            """), transforms={"salary": int, "commission": float})

        sales_employees = employees.where(dept="Sales").sort("salary desc,commission")

        sales_employees.present()
        print(list(sales_employees.all.emp_id))

        with self.subTest():
            self.assertEqual(['0005', '0001', '0004', '0003'],
                             list(sales_employees.all.emp_id))

    def test_unique(self):
        test_size = 10
        t1 = make_test_table(self.make_data_object, test_size)

        t2 = t1.unique()
        with self.subTest():
            self.assertEqual(len(t1), len(t2))

        t3 = t1.unique(key=lambda rec: rec.c)
        with self.subTest():
            self.assertEqual(test_size, len(t3))


@make_test_classes
class TableOutputTests:
    def test_basic_present(self):
        if rich is None:
            import warnings
            warnings.warn("rich not installed, cannot run test")
            return

        from rich import box
        from io import StringIO
        table = lt.Table().csv_import(textwrap.dedent("""\
            a,b
            10,100
            20,200
            """))
        table.present()
        out = StringIO()
        table.present(file=out, box=box.ASCII)
        expected = textwrap.dedent("""\
            +----------+
            | A  | B   |
            |----+-----|
            | 10 | 100 |
            | 20 | 200 |
            +----------+
            """)
        with self.subTest():
            self.assertEqual(expected, out.getvalue())

        # test bugfix when table has attribute "default"
        table = lt.Table().csv_import(textwrap.dedent("""\
            a,b,default
            10,100,purple
            15,150,
            20,200,orange
            """))
        table.present()
        out = StringIO()
        table.present(file=out, box=box.ASCII)
        expected = textwrap.dedent("""\
            +--------------------+
            | A  | B   | Default |
            |----+-----+---------|
            | 10 | 100 | purple  |
            | 15 | 150 |         |
            | 20 | 200 | orange  |
            +--------------------+
            """)
        with self.subTest():
            self.assertEqual(expected, out.getvalue())

        # test groupby
        table = lt.Table().csv_import(textwrap.dedent("""\
            a,b,default
            10,100,purple
            15,150,purple
            20,200,orange
            """))
        table.present()
        table.present(box=box.ASCII, groupby="default")
        out = StringIO()
        table.present(file=out, box=box.ASCII, groupby="default")
        expected = textwrap.dedent("""\
            +--------------------+
            | A  | B   | Default |
            |----+-----+---------|
            | 10 | 100 | purple  |
            | 15 | 150 |         |
            | 20 | 200 | orange  |
            +--------------------+
            """)
        with self.subTest():
            self.assertEqual(expected, out.getvalue())

        table = lt.Table().csv_import(textwrap.dedent("""\
            a,b,default
            10,100,purple
            15,150,purple
            15,200,orange
            15,250,orange
            20,250,orange
            """))
        table.present()
        table.present(box=box.ASCII, groupby="default a")
        out = StringIO()
        table.present(file=out, box=box.ASCII, groupby="default a")
        expected = textwrap.dedent("""\
            +--------------------+
            | A  | B   | Default |
            |----+-----+---------|
            | 10 | 100 | purple  |
            | 15 | 150 |         |
            | 15 | 200 | orange  |
            |    | 250 |         |
            | 20 | 250 |         |
            +--------------------+
            """)
        with self.subTest():
            self.assertEqual(expected, out.getvalue())

        table = lt.Table().csv_import(textwrap.dedent("""\
            a,b,default
            10,100,purple
            15,200,orange
            15,150,purple
            20,250,orange
            15,250,orange
            """))
        table.sort("default desc,a")
        table.present()
        table.present(box=box.ASCII, groupby="default a")
        out = StringIO()
        table.present(file=out, box=box.ASCII, groupby="default a")
        expected = textwrap.dedent("""\
            +--------------------+
            | A  | B   | Default |
            |----+-----+---------|
            | 10 | 100 | purple  |
            | 15 | 150 |         |
            | 15 | 200 | orange  |
            |    | 250 |         |
            | 20 | 250 |         |
            +--------------------+
            """)
        with self.subTest():
            self.assertEqual(expected, out.getvalue())

    def test_markdown(self):
        table = lt.Table().csv_import(textwrap.dedent("""\
            a,b
            10,100
            20,200
            """))
        out_markdown = table.as_markdown()
        print(out_markdown)
        expected = textwrap.dedent("""\
            | a | b |
            |---|---|
            | 10 | 100 |
            | 20 | 200 |
            """)
        with self.subTest():
            self.assertEqual(expected, out_markdown)

        # test bugfix when table has attribute "default"
        table = lt.Table().csv_import(textwrap.dedent("""\
            a,b,default
            10,100,purple
            15,150,
            20,200,orange
            """))
        out_markdown = table.as_markdown()
        print(out_markdown)
        expected = textwrap.dedent("""\
            | a | b | default |
            |---|---|---|
            | 10 | 100 | purple |
            | 15 | 150 |  |
            | 20 | 200 | orange |
            """)
        with self.subTest():
            self.assertEqual(expected, out_markdown)

        # test grouping in as_markdown
        table = lt.Table().csv_import(textwrap.dedent("""\
            a,b,default
            10,100,purple
            15,150,purple
            20,200,orange
            """))
        out_markdown = table.as_markdown(groupby="default")
        print(out_markdown)
        expected = textwrap.dedent("""\
            | a | b | default |
            |---|---|---|
            | 10 | 100 | purple |
            | 15 | 150 |  |
            | 20 | 200 | orange |
            """)
        with self.subTest():
            self.assertEqual(expected, out_markdown)

        table = lt.Table().csv_import(textwrap.dedent("""\
            a,b,default
            10,100,purple
            15,200,orange
            15,150,purple
            20,250,orange
            15,250,orange
            """))
        table.sort("default desc,a")
        out_markdown = table.as_markdown(groupby="default a")
        print(out_markdown)
        expected = textwrap.dedent("""\
            | a | b | default |
            |---|---|---|
            | 10 | 100 | purple |
            | 15 | 150 |  |
            | 15 | 200 | orange |
            |  | 250 |  |
            | 20 | 250 |  |
            """)
        with self.subTest():
            self.assertEqual(expected, out_markdown)

# sample import data sets
csv_data = """\
a,b,c
0,0,0
0,0,1
0,0,2
0,1,0
0,1,1
0,1,2
0,2,0
0,2,1
0,2,2
1,0,0
1,0,1
1,0,2
1,1,0
1,1,1
1,1,2
1,2,0
1,2,1
1,2,2
2,0,0
2,0,1
2,0,2
2,1,0
2,1,1
2,1,2
2,2,0
2,2,1
2,2,2

"""

json_data = """\
    {"a": 0, "b": 0, "c": 0}
    {"a": 0, "b": 0, "c": 1}
    {"a": 0, "b": 0, "c": 2}
    {"a": 0, "b": 1, "c": 0}
    {"a": 0, "b": 1, "c": 1}
    {"a": 0, "b": 1, "c": 2}
    {"a": 0, "b": 2, "c": 0}
    {"a": 0, "b": 2, "c": 1}
    {"a": 0, "b": 2, "c": 2}
    {"a": 1, "b": 0, "c": 0}
    {"a": 1, "b": 0, "c": 1}
    {"a": 1, "b": 0, "c": 2}
    {"a": 1, "b": 1, "c": 0}
    {"a": 1, "b": 1, "c": 1}
    {"a": 1, "b": 1, "c": 2}
    {"a": 1, "b": 2, "c": 0}
    {"a": 1, "b": 2, "c": 1}
    {"a": 1, "b": 2, "c": 2}
    {"a": 2, "b": 0, "c": 0}
    {"a": 2, "b": 0, "c": 1}
    {"a": 2, "b": 0, "c": 2}
    {"a": 2, "b": 1, "c": 0}
    {"a": 2, "b": 1, "c": 1}
    {"a": 2, "b": 1, "c": 2}
    {"a": 2, "b": 2, "c": 0}
    {"a": 2, "b": 2, "c": 1}
    {"a": 2, "b": 2, "c": 2}

"""

fixed_width_data = """\
0 0 0
0 0 1
0 0 2
0 1 0
0 1 1
0 1 2
0 2 0
0 2 1
0 2 2
1 0 0
1 0 1
1 0 2
1 1 0
1 1 1
1 1 2
1 2 0
1 2 1
1 2 2
2 0 0
2 0 1
2 0 2
2 1 0
2 1 1
2 1 2
2 2 0
2 2 1
2 2 2

"""


@make_test_classes
class TableImportExportTests:
    def test_csv_export(self):
        from itertools import permutations
        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)
        for fieldnames in permutations(list('abc')):
            out = io.StringIO()
            t1.csv_export(out, fieldnames)
            out.seek(0)
            outlines = out.read().splitlines()
            out.close()
            with self.subTest():
                self.assertEqual(','.join(fieldnames), outlines[0])
            with self.subTest():
                self.assertEqual(test_size**3+1, len(outlines))
            for ob, line in zip(t1, outlines[1:]):
                csv_vals = line.split(',')
                with self.subTest(ob=ob, csv_vals=csv_vals):
                    self.assertTrue(all(
                        int(csv_vals[i]) == getattr(ob, fld) for i, fld in enumerate(fieldnames)
                    ))

        # rerun using an empty table
        t1 = lt.Table()
        for fieldnames in permutations(list('abc')):
            out = io.StringIO()
            t1.csv_export(out, fieldnames)
            out.seek(0)
            outlines = out.read().splitlines()
            out.close()
            with self.subTest():
                self.assertEqual(','.join(fieldnames), outlines[0])
            with self.subTest():
                self.assertEqual(1, len(outlines))

        # rerun using an empty table, with indexes to dictate fieldnames
        for fieldnames in permutations(list('abc')):
            t1 = lt.Table()
            for fld in fieldnames:
                t1.create_index(fld)
            out = io.StringIO()
            t1.csv_export(out)
            out.seek(0)
            outlines = out.read().splitlines()
            out.close()
            with self.subTest():
                self.assertEqual(set(fieldnames), set(outlines[0].split(',')))
            with self.subTest():
                self.assertEqual(1, len(outlines))

    def test_csv_export_to_string(self):
        from itertools import permutations
        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)
        for fieldnames in permutations(list('abc')):
            out_string = t1.csv_export(None, fieldnames)
            outlines = out_string.splitlines()
            with self.subTest():
                self.assertEqual(','.join(fieldnames), outlines[0])
            with self.subTest():
                self.assertEqual(test_size ** 3 + 1, len(outlines))
            for ob, line in zip(t1, outlines[1:]):
                csv_vals = line.split(',')
                with self.subTest(ob=ob, csv_vals=csv_vals):
                    self.assertTrue(all(
                        int(csv_vals[i]) == getattr(ob, fld) for i, fld in enumerate(fieldnames)
                    ))

    def test_csv_import(self):
        data = csv_data
        incsv = io.StringIO(data)
        csvtable = lt.Table().csv_import(incsv, transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        with self.subTest():
            self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, csvtable)))
        with self.subTest():
            self.assertEqual(sum(1 for line in data.splitlines() if line.strip())-1, len(csvtable))

        incsv = io.StringIO(data)
        row_prototype = self.make_data_object(0, 0, 0)
        csvtable2 = lt.Table().csv_import(incsv, transforms={'a': int, 'b': int, 'c': int}, row_class=type(row_prototype))[:3]

        print(type(t1[0]).__name__, t1[0])
        print(type(csvtable2[0]).__name__, csvtable2[0])
        with self.subTest():
            self.assertEqual(type(t1[0]), type(csvtable2[0]))

    def test_csv_compressed_import(self):
        tt = lt.Table().csv_import("test/abc.csv", transforms=dict.fromkeys("abc", int))
        print("abc.csv", tt.info())

        compressed_files = [
            "abc.csv.zip",
            "abc.csv.gz",
            "abc.csv.xz",
        ]
        for name in compressed_files:
            import_source_name = "test/" + name
            tt2 = lt.Table().csv_import(import_source_name, transforms=dict.fromkeys("abc", int))
            print(name, tt2.info())
            expected_info = {**tt.info(), "name": import_source_name}
            with self.subTest(name=name):
                self.assertEqual(expected_info, tt2.info())
            with self.subTest(name=name):
                self.assertEqual(sum(tt.all.a), sum(tt2.all.a))
            with self.subTest(name=name):
                self.assertEqual(sum(tt.all.b), sum(tt2.all.b))
            with self.subTest(name=name):
                self.assertEqual(sum(tt.all.c), sum(tt2.all.c))

        # test separately, no transforms for JSON imports
        import_source_name = "test/abc.json.gz"
        tt2 = lt.Table().json_import("test/abc.json.gz", streaming=True)
        print("abc.json.gz", tt2.info())
        expected_info = {**tt.info(), "name": import_source_name}
        with self.subTest():
            self.assertEqual(expected_info, tt2.info())
        with self.subTest():
            self.assertEqual(sum(tt.all.a), sum(tt2.all.a))
        with self.subTest():
            self.assertEqual(sum(tt.all.b), sum(tt2.all.b))
        with self.subTest():
            self.assertEqual(sum(tt.all.c), sum(tt2.all.c))

    def test_csv_import_source_info(self):
        imports = [
            ("abc.csv", lt.ImportSourceType.file),
            ("abc.tsv", lt.ImportSourceType.file),
            ("abc.xlsx", lt.ImportSourceType.file),
            ("abc.csv.zip", lt.ImportSourceType.zip),
            ("abc.csv.gz", lt.ImportSourceType.gzip),
            ("abc.csv.xz", lt.ImportSourceType.lzma),
            ("a,b,c\n1,2,3", lt.ImportSourceType.string),
        ]
        for fname, expected_type in imports:
            if "\n" not in fname:
                import_name = "test/" + fname
            else:
                import_name = fname
            if import_name.endswith(".csv"):
                tbl = lt.Table().csv_import(import_name)
            elif import_name.endswith(".xlsx"):
                tbl = lt.Table().excel_import(import_name)
            else:
                tbl = lt.Table().tsv_import(import_name)

            print(repr(import_name), tbl.import_source, tbl.import_source_type)

            if "\n" not in fname:
                with self.subTest():
                    self.assertEqual(import_name, tbl.import_source)
            else:
                with self.subTest():
                    self.assertEqual(None, tbl.import_source)
            with self.subTest():
                self.assertEqual(expected_type, tbl.import_source_type)

    def test_csv_import_from_url(self):
        from http.server import HTTPServer, BaseHTTPRequestHandler
        from http import HTTPStatus
        import threading
        import time
        import urllib.error
        import urllib.request

        if SKIP_CSV_IMPORT_USING_URL_TESTS:
            self.skipTest("CSV import tests skipped")

        class CSVTestRequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.log_message("received %s %s", self.command, self.path)
                path = self.path
                if path == "/EXIT":
                    self.send_response(HTTPStatus.OK)
                    self.end_headers()
                    threading.Thread(target=lambda: self.server.shutdown()).start()

                elif path == "/":
                    self.send_response(HTTPStatus.OK)
                    self.end_headers()

                elif path.startswith("/abc.csv"):
                    send_bytes = b"a,b,c\n1,2,3\n"
                    self.send_response(HTTPStatus.OK)
                    self.end_headers()
                    self.wfile.write(send_bytes)

        def run(server_class=HTTPServer, handler_class=CSVTestRequestHandler):
            server_address = ('', 8888)
            httpd = server_class(server_address, handler_class)
            httpd.serve_forever()

        def run_background_test_server():
            p = threading.Thread(target=run)
            p.start()

            for tries_remaining in reversed(range(20)):
                try:
                    with urllib.request.urlopen("http://localhost:8888/"):
                        break
                except urllib.error.URLError:
                    if tries_remaining:
                        time.sleep(0.25)

            return p

        web_address = "http://localhost:8888"
        p = run_background_test_server()

        url = web_address + "/abc.csv"
        try:
            tbl = lt.Table().csv_import(url)
        finally:
            with urllib.request.urlopen(web_address + "/EXIT"):
                pass
            p.join()

        tbl.present()
        with self.subTest():
            self.assertEqual(url, tbl.import_source)
        with self.subTest():
            self.assertEqual(lt.ImportSourceType.url, tbl.import_source_type)

    def test_csv_filtered_import(self):
        test_size = 3
        tt = lt.Table().csv_import("test/abc.csv", transforms=dict.fromkeys("abc", int))
        print("abc.csv", tt.info())

        tt = lt.Table().csv_import("test/abc.csv", transforms=dict.fromkeys("abc", int),
                                   filters={"c": lt.Table.eq(1)})
        print(tt.info())
        with self.subTest():
            self.assertEqual(test_size * test_size, len(tt))

        tt = lt.Table().csv_import("test/abc.csv", transforms=dict.fromkeys("abc", int),
                                   filters={"c": 1})
        print(tt.info())
        with self.subTest():
            self.assertEqual(test_size * test_size, len(tt))

        tt = lt.Table().csv_import("test/abc.csv", transforms=dict.fromkeys("abc", int),
                                   filters={"c": lambda x: 0 < x < 2})
        print(tt.info())
        with self.subTest():
            self.assertEqual(test_size * test_size, len(tt))

        # test all special comparators when used as filters
        #     is_none - attribute value is None
        #     is_not_none - attribute value is not None
        #     is_null - attribute value is None, "", or not defined
        #     is_not_null - attribute value is defined, and is not None or ""
        #     startswith - attribute value starts with a given string
        #     endswith - attribute value ends with a given string
        #     re_match - attribute value matches a regular expression

        print()
        input_data = textwrap.dedent("""\
        name,a,b,c
        "A",100,100,100
        "B",200,,200
        "A1",101,101,101
        "B1",201,,201
        "C1",301,,301
        ,99,99,99
        """)
        lt.Table().csv_import(input_data,
                              transforms=dict.fromkeys("abc", int),
                              ).present()

        """
        +-------------------------+
        | Name |   A |    B |   C |
        |------+-----+------+-----|
        | A    | 100 |  100 | 100 |
        | B    | 200 | None | 200 |
        | A1   | 101 |  101 | 101 |
        | B1   | 201 | None | 201 |
        | C1   | 301 | None | 301 |
        |      |  99 |   99 |  99 |
        +-------------------------+
        """

        print("is_none()")
        x = lt.Table().csv_import(input_data,
                                         transforms=dict.fromkeys("abc", int),
                                         filters={"b": lt.Table.is_none()})
        with self.subTest():
            self.assertEqual(3, len(x))
        with self.subTest():
            self.assertTrue(all(b is None for b in x.all.b))

        print("is_not_none()")
        x = lt.Table().csv_import(input_data,
                                  transforms=dict.fromkeys("abc", int),
                                  filters={"b": lt.Table.is_not_none()})
        with self.subTest():
            self.assertEqual(3, len(x))
        with self.subTest():
            self.assertEqual(300, sum(x.all.b))

        print("b is_null()")
        x = lt.Table().csv_import(input_data,
                                  transforms=dict.fromkeys("abc", int),
                                  filters={"b": lt.Table.is_null()})
        with self.subTest():
            self.assertEqual(3, len(x))
        with self.subTest():
            self.assertTrue(all(b is None for b in x.all.b))

        print("b is_not_null()")
        x = lt.Table().csv_import(input_data,
                                  transforms=dict.fromkeys("abc", int),
                                  filters={"b": lt.Table.is_not_null()})
        with self.subTest():
            self.assertEqual(3, len(x))
        with self.subTest():
            self.assertEqual(300, sum(x.all.b))

        print("name is_null()")
        x = lt.Table().csv_import(input_data,
                                  transforms=dict.fromkeys("abc", int),
                                  filters={"name": lt.Table.is_null()})
        with self.subTest():
            self.assertEqual(1, len(x))
        with self.subTest():
            self.assertEqual(3*99, x[0].a + x[0].b + x[0].c)

        print("name is_not_null()")
        x = lt.Table().csv_import(input_data,
                                  transforms=dict.fromkeys("abc", int),
                                  filters={"name": lt.Table.is_not_null()})
        with self.subTest():
            self.assertEqual(5, len(x))
        with self.subTest():
            self.assertEqual("A B A1 B1 C1".split(), list(x.all.name))

        print("name startswith('B')")
        x = lt.Table().csv_import(input_data,
                                  transforms=dict.fromkeys("abc", int),
                                  filters={"name": lt.Table.startswith("B")})
        with self.subTest():
            self.assertEqual(2, len(x))
        with self.subTest():
            self.assertEqual("B B1".split(), list(x.all.name))

        print("name endswith('1')")
        x = lt.Table().csv_import(input_data,
                                  transforms=dict.fromkeys("abc", int),
                                  filters={"name": lt.Table.endswith("1")})
        with self.subTest():
            self.assertEqual(3, len(x))
        with self.subTest():
            self.assertEqual("A1 B1 C1".split(), list(x.all.name))

        print(r"name re_match(r'[AB]\d')")
        x = lt.Table().csv_import(input_data,
                                  transforms=dict.fromkeys("abc", int),
                                  filters={"name": lt.Table.re_match(r"[AB]\d")})
        with self.subTest():
            self.assertEqual(2, len(x))
        with self.subTest():
            self.assertEqual("A1 B1".split(), list(x.all.name))

    def test_csv_string_import(self):
        data = csv_data
        csvtable = lt.Table().csv_import(csv_source=data, transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        with self.subTest():
            self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, csvtable)))
        with self.subTest():
            self.assertEqual(sum(1 for line in data.splitlines() if line.strip())-1, len(csvtable))

        row_prototype = self.make_data_object(0, 0, 0)
        csvtable2 = lt.Table().csv_import(data, transforms={'a': int, 'b': int, 'c': int},
                                          row_class=type(row_prototype))[:3]

        print(type(t1[0]).__name__, t1[0])
        print(type(csvtable2[0]).__name__, csvtable2[0])
        with self.subTest():
            self.assertEqual(type(t1[0]), type(csvtable2[0]))

    def test_csv_limit_import(self):
        data = csv_data
        import_limit = 10
        csvtable = lt.Table().csv_import(csv_source=data, transforms={'a': int, 'b': int, 'c': int},
                                         limit=import_limit)

        with self.subTest():
            self.assertEqual(import_limit, len(csvtable))

        csvtable = lt.Table().csv_import(csv_source=data, transforms={'a': int, 'b': int, 'c': int},
                                         limit=0)

        with self.subTest():
            self.assertEqual(0, len(csvtable))

    def test_csv_string_list_import(self):
        data = csv_data
        csvtable = lt.Table().csv_import(csv_source=data.splitlines(), transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        with self.subTest():
            self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, csvtable)))
        with self.subTest():
            self.assertEqual(sum(1 for line in data.splitlines() if line.strip())-1, len(csvtable))

        row_prototype = self.make_data_object(0, 0, 0)
        csvtable2 = lt.Table().csv_import(data, transforms={'a': int, 'b': int, 'c': int},
                                          row_class=type(row_prototype))[:3]

        print(type(t1[0]).__name__, t1[0])
        print(type(csvtable2[0]).__name__, csvtable2[0])
        with self.subTest():
            self.assertEqual(type(t1[0]), type(csvtable2[0]))

    def test_csv_numeric_transforms(self):
        data = textwrap.dedent("""\
            type,value
            int,1000
            float,3.14
            empty,
            str,ⓠ*bert
            """)

        with self.subTest("convert_numeric"):
            tbl = lt.Table().csv_import(data, transforms={'value': lt.Table.convert_numeric})
            tbl.present()
            self.assertEqual([1000, 3.14, '', 'ⓠ*bert'], list(tbl.all.value))

        with self.subTest("convert_numeric()"):
            tbl = lt.Table().csv_import(data, transforms={'value': lt.Table.convert_numeric()})
            tbl.present()
            self.assertEqual([1000, 3.14, '', 'ⓠ*bert'], list(tbl.all.value))

        with self.subTest("convert_numeric(non_numeric=None)"):
            tbl = lt.Table().csv_import(data, transforms={'value': lt.Table.convert_numeric(non_numeric=None)})
            tbl.present()
            self.assertEqual([1000, 3.14, '', None], list(tbl.all.value))

        with self.subTest("convert_numeric(non_numeric=0)"):
            tbl = lt.Table().csv_import(data, transforms={'value': lt.Table.convert_numeric(non_numeric=0)})
            tbl.present()
            self.assertEqual([1000, 3.14, '', 0], list(tbl.all.value))

        with self.subTest("convert_numeric(int_to_float=True)"):
            tbl = lt.Table().csv_import(data, transforms={'value': lt.Table.convert_numeric(force_float=True)})
            tbl.present()
            self.assertEqual([1000.0, 3.14, '', 'ⓠ*bert'], list(tbl.all.value))
            self.assertEqual([float, float, str, str], list(type(v) for v in tbl.all.value))

        with self.subTest("convert_numeric(non_numeric=None, empty=None)"):
            tbl = lt.Table().csv_import(data, transforms={'value': lt.Table.convert_numeric(non_numeric=None, empty=None)})
            tbl.present()
            self.assertEqual([1000, 3.14, None, None], list(tbl.all.value))

        with self.subTest("convert_numeric(empty=None)"):
            tbl = lt.Table().csv_import(data, transforms={'value': lt.Table.convert_numeric(empty=None)})
            tbl.present()
            self.assertEqual([1000, 3.14, None, 'ⓠ*bert'], list(tbl.all.value))

    def test_json_export_streaming(self):
        from itertools import permutations
        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)
        for fieldnames in permutations(list('abc')):
            out_string = t1.json_export(None, fieldnames=fieldnames, streaming=True)
            outlines = out_string.splitlines()

            with self.subTest(fieldnames=fieldnames):
                self.assertEqual(test_size**3, len(outlines))

            for ob, line in zip(t1, outlines):
                json_dict = json.loads(line)
                t1_dataobj = make_dataobject_from_ob(ob)
                with self.subTest(ob=ob, line=line):
                    self.assertEqual(t1_dataobj, lt.DataObject(**json_dict))

    def test_json_export_nonstreaming(self):
        from itertools import permutations
        import json
        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)
        for fieldnames in permutations(list('abc')):
            out_string = t1.json_export(None, fieldnames=fieldnames, streaming=False)
            observed_json = json.loads(out_string)

            with self.subTest(fieldnames=fieldnames):
                self.assertEqual(test_size**3, len(observed_json))

            for ob, json_dict in zip(t1, observed_json):
                t1_dataobj = make_dataobject_from_ob(ob)
                with self.subTest(ob=ob, json_dict=json_dict):
                    self.assertEqual(t1_dataobj, lt.DataObject(**json_dict))

    def test_json_import(self):
        data = json_data
        injson = io.StringIO(data)
        jsontable = lt.Table().json_import(injson, streaming=True, transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        with self.subTest():
            self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, jsontable)))
        with self.subTest():
            self.assertEqual(len([d for d in data.splitlines() if d.strip()]), len(jsontable))

    def test_json_string_import(self):
        data = json_data
        jsontable = lt.Table().json_import(data, streaming=True, transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        with self.subTest():
            self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, jsontable)))
        with self.subTest():
            self.assertEqual(len([d for d in data.splitlines() if d.strip()]), len(jsontable))

    def test_json_string_list_import(self):
        data = json_data
        jsontable = lt.Table().json_import(data.splitlines(), streaming=True, transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        with self.subTest():
            self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, jsontable)))
        with self.subTest():
            self.assertEqual(len([d for d in data.splitlines() if d.strip()]), len(jsontable))

    def test_json_nonstreaming_with_path_import(self):
        data = json_data
        data = ',\n'.join(data.rstrip().splitlines())
        json_input0 = '[' + data + ']'
        json_input1 = '{ "data": [' + data + ']}'
        json_input2 = '{ "data": { "items": [' + data + ']}}'

        for json_input, path in [
            (json_input0, ""),
            (json_input1, "data"),
            (json_input2, "data.items"),
        ]:
            jsontable = lt.Table().json_import(json_input,
                                               path=path,
                                               transforms={'a': int, 'b': int, 'c': int})

            test_size = 3
            t1 = make_test_table(self.make_data_object, test_size)

            with self.subTest():
                self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, jsontable)))
            with self.subTest():
                self.assertEqual(len([d for d in data.splitlines() if d.strip()]), len(jsontable))

    def test_fixed_width_import(self):
        data = fixed_width_data
        data_file = io.StringIO(data)
        fw_spec = [('a', 0, None, int), ('b', 2, None, int), ('c', 4, None, int), ]
        tt = lt.Table().insert_many(lt.DataObject(**rec) for rec in lt.FixedWidthReader(fw_spec, data_file))

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        with self.subTest():
            self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, tt)))
        with self.subTest():
            self.assertEqual(len([d for d in data.splitlines() if d.strip()]), len(tt))

    def test_fixed_width_string_import(self):
        data = fixed_width_data
        fw_spec = [('a', 0, None, int), ('b', 2, None, int), ('c', 4, None, int), ]
        tt = lt.Table().insert_many(lt.DataObject(**rec) for rec in lt.FixedWidthReader(fw_spec, data))

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        with self.subTest():
            self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, tt)))
        with self.subTest():
            self.assertEqual(len([d for d in data.splitlines() if d.strip()]), len(tt))

    def test_fixed_width_string_list_import(self):
        data = fixed_width_data
        fw_spec = [('a', 0, None, int), ('b', 2, None, int), ('c', 4, None, int),]
        tt = lt.Table().insert_many(lt.DataObject(**rec) for rec in lt.FixedWidthReader(fw_spec, data.splitlines()))

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        with self.subTest():
            self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, tt)))
        with self.subTest():
            self.assertEqual(len([d for d in data.splitlines() if d.strip()]), len(tt))

    def test_excel_import(self):
        file_name = "test/abc.xlsx"
        excel_table = lt.Table().excel_import(file_name, transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        with self.subTest():
            self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, excel_table)))
        with self.subTest():
            self.assertEqual(sum(1 for line in csv_data.splitlines() if line.strip())-1, len(excel_table))

        row_prototype = self.make_data_object(0, 0, 0)
        csvtable2 = lt.Table().excel_import(
            file_name, transforms={'a': int, 'b': int, 'c': int}, row_class=type(row_prototype)
        )[:3]

        print(type(t1[0]).__name__, t1[0])
        print(type(csvtable2[0]).__name__, csvtable2[0])
        with self.subTest():
            self.assertEqual(type(t1[0]), type(csvtable2[0]))

    def test_excel_export(self):
        file_name = "test/abc.xlsx"
        excel_table = lt.Table().excel_import(file_name, transforms={'a': int, 'b': int, 'c': int}, limit=1)
        outfile = io.BytesIO()
        excel_table.excel_export(outfile)
        exported_table = lt.Table().excel_import(outfile, transforms={'a': int, 'b': int, 'c': int})

        with self.subTest():
            self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(exported_table, excel_table)))


@make_test_classes
class TablePivotTests:
    def test_pivot(self):
        test_size = 5
        t1 = make_test_table(self.make_data_object, test_size)
        t1.create_index('a')
        t1.create_index('b')
        t1pivot = t1.pivot('a')
        t1pivot.dump()
        t1pivot.dump_counts()
        t1pivot.summary_counts()

        t2pivot = t1.pivot('a b')
        t2pivot.dump()
        t2pivot.dump_counts()
        t2pivot.summary_counts()

        # TODO - add asserts


class TableSearchTests(unittest.TestCase):
    recipe_data = textwrap.dedent("""\
        id,title,ingredients
        1,Tuna casserole,"tuna, noodles, Cream of Mushroom Soup"
        2,Hawaiian pizza,pizza dough pineapple ham tomato sauce
        3,Margherita pizza,pizza dough cheese pesto artichoke hearts
        4,Pepperoni pizza,pizza dough cheese tomato sauce pepperoni
        5,Grilled cheese sandwich,bread cheese butter
        6,Tuna melt,tuna mayonnaise tomato bread cheese
        7,Chili dog,hot dog chili onion bun
        8,French toast,egg milk vanilla bread maple syrup
        9,BLT,bread bacon lettuce tomato mayonnaise
        10,Reuben sandwich,rye bread sauerkraut corned beef swiss cheese russian dressing thousand island
        11,Hamburger,ground beef bun lettuce ketchup mustard pickle
        12,Cheeseburger,ground beef bun lettuce ketchup mustard pickle cheese
        13,Bacon cheeseburger,ground beef bun lettuce ketchup mustard pickle cheese bacon
        """)

    def setUp(self):
        self.recipes = lt.Table().csv_import(self.recipe_data, transforms=dict(id=int))
        self.recipes.create_search_index("ingredients")

    @announce_test
    def test_access_non_existent_search_attribute(self):
        with self.assertRaises(ValueError, msg="failed to raise ValueError when accessing non-existent search index"):
            self.recipes.search.title("xyz")

    @announce_test
    def test_search_dir(self):
        self.assertEqual(['ingredients'], dir(self.recipes.search), "failed to generate correct dir() response")

    @announce_test
    def test_text_search(self):
        for query, expected in [
            ("", []),
            ("tuna", [1, 6]),
            ("tuna +cheese", [6, 3, 4, 5, 10, 12, 13, 1]),
            ("pineapple +bacon lettuce beef -sauerkraut tomato", [9, 13, 2, 11, 12, 4, 6, 10]),
            ("pizza dough -pineapple", [3, 4, 2]),
            ("pizza dough --pineapple", [3, 4]),
            ("bread bacon", [9, 5, 6, 8, 10, 13]),
            ("bread ++bacon", [9, 13]),
            ("bread ++anchovies", []),
            ("bread ++bacon ++anchovies", []),
            ("bread bacon --anchovies", [9, 5, 6, 8, 10, 13]),
        ]:
            matches = self.recipes.search.ingredients(query, min_score=-10000)
            match_ids = [recipe.id for recipe, _ in matches]
            print(repr(query), '->', [(recipe.id, score) for recipe, score in matches])
            with self.subTest(query=query):
                self.assertEqual(expected, match_ids,
                                 "invalid results for query {!r}, expected {}, got {}".format(query, expected, match_ids))

    @announce_test
    def test_invalidate_index(self):
        self.recipes.pop(0)
        with self.assertRaises(lt.SearchIndexInconsistentError,
                               msg="failed to raise exception when searching modified table"):
            self.recipes.search.ingredients("bacon")

    @announce_test
    def test_search_with_keywords(self):
        for query, expected, expected_words in [
                ("tuna", [1, 6], [{'tuna', 'noodles', 'cream', 'of', 'mushroom', 'soup'},
                                  {'tuna', 'mayonnaise', 'tomato', 'bread', 'cheese'}]),
                ]:
            matches = self.recipes.search.ingredients(query, min_score=-10000, include_words=True)
            match_ids = [recipe.id for recipe, score, words in matches]
            print(repr(query), '->', [(recipe.id, score, words) for recipe, score, words in matches])
            with self.subTest():
                self.assertEqual(expected, match_ids,
                                 "invalid results for query {!r}, expected {}, got {}".format(query, expected, match_ids))
            match_words = [set(words) for recipe, score, words in matches]
            with self.subTest():
                self.assertEqual(expected_words, match_words,
                                 "invalid match words for query {!r}, expected {}, got {}".format(query,
                                                                                                  expected_words,
                                                                                                  match_words))

    @announce_test
    def test_search_with_limit(self):
        for query, expected in [
            ("", []),
            ("tuna", [1, 6]),
            ("tuna +cheese", [6, 3, 4]),
            ("pineapple +bacon lettuce beef -sauerkraut tomato", [9, 13, 2]),
            ("pizza dough -pineapple", [3, 4, 2]),
            ("pizza dough --pineapple", [3, 4]),
            ("bread bacon", [9, 5, 6]),
            ("bread ++bacon", [9, 13]),
            ("bread ++anchovies", []),
            ("bread ++bacon ++anchovies", []),
            ("bread bacon --anchovies", [9, 5, 6]),
        ]:
            matches = self.recipes.search.ingredients(query, min_score=-10000, limit=3)
            match_ids = [recipe.id for recipe, _ in matches]
            print(repr(query), '->', [(recipe.id, score) for recipe, score in matches])
            with self.subTest(query=query):
                self.assertEqual(expected, match_ids,
                                 "invalid results for query {!r}, expected {}, got {}".format(query, expected, match_ids))

    @announce_test
    def test_search_with_min_score(self):
        for query, expected in [
            ("", []),
            ("tuna", []),
            ("tuna +cheese", [6,]),
            ("pineapple +bacon lettuce beef -sauerkraut tomato", [9, 13]),
            ("pizza dough -pineapple", []),
            ("pizza dough --pineapple", []),
            ("bread bacon", []),
            ("bread ++bacon", [9,]),
            ("bread ++anchovies", []),
            ("bread ++bacon ++anchovies", []),
            ("bread bacon --anchovies", []),
        ]:
            matches = self.recipes.search.ingredients(query, min_score=1000)
            match_ids = [recipe.id for recipe, _ in matches]
            print(repr(query), '->', [(recipe.id, score) for recipe, score in matches])
            with self.subTest(query=query):
                self.assertEqual(expected, match_ids,
                                 "invalid results for query {!r}, expected {}, got {}".format(query, expected, match_ids))


class TableSearchTests_DataObjects(TableSearchTests, UsingDataObjects):
    pass

class TableSearchTests_Namedtuples(TableSearchTests, UsingNamedtuples):
    pass

class TableSearchTests_DataNamedtuples(TableSearchTests, UsingDataNamedtuples):
    pass

class TableSearchTests_Slotted(TableSearchTests, UsingSlottedObjects):
    pass

class TableSearchTests_SimpleNamespace(TableSearchTests, UsingSimpleNamespace):
    pass

if dataclasses is not None:
    class TableSearchTests_Dataclasses(TableSearchTests, UsingDataclasses):
        pass

if pydantic is not None:
    class TableSearchTests_PydanticModels(TableSearchTests, UsingPydanticModel):
        pass

    class TableSearchTests_PydanticImmutableModels(TableSearchTests, UsingPydanticImmutableModel):
        pass

    class TableSearchTests_PydanticORMModels(TableSearchTests, UsingPydanticORMModel):
        pass

if attr is not None:
    class TableSearchTests_AttrClasses(TableSearchTests, UsingAttrClass):
        pass

if traitlets is not None:
    class TableSearchTests_TraitletsClasses(TableSearchTests, UsingTraitletsClass):
        pass

if SlottedWithDict is not None:
    class TableSearchTests_SlottedWithDict(TableSearchTests, UsingSlottedWithDictObjects):
        pass


class InitialTest(unittest.TestCase):
    from littletable import (
        __version__ as littletable_version,
        __version_time__ as littletable_version_time,
        __version_info__ as littletable_version_info,
    )

    print(
        f"Beginning test of littletable, version {littletable_version}, {littletable_version_time}",
    )
    print(littletable_version_info)
    print("Python version", sys.version)
    print()


if __name__ == '__main__':

    unittest.main()
