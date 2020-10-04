#
# unit_tests.py
#
# unit tests for littletable library
#

import unittest
import littletable as lt
import itertools
import json
from collections import namedtuple
from operator import attrgetter
import textwrap
try:
    from types import SimpleNamespace
except ImportError:
    class SimpleNamespace(object):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def __repr__(self):
            keys = sorted(self.__dict__)
            items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
            return "{}({})".format(type(self).__name__, ", ".join(items))
        def __eq__(self, other):
            return vars(self) == vars(other)

try:
    import dataclasses
except ImportError:
    # pre Py3.7 (or 3.6 with backported dataclasses)
    dataclasses = None
else:
    # must be wrapped in exec, since this syntax is not legal in earlier Pythons
    exec("""\
@dataclasses.dataclass
class DataDataclass:
    a: int
    b: int
    c: int
""")

DataTuple = namedtuple("DataTuple", "a b c")

import sys
PY_2 = sys.version_info[0] == 2
PY_3 = sys.version_info[0] == 3

if PY_3:
    import io
else:
    import StringIO as io

class Slotted(object):
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


class TestDataObjects(unittest.TestCase):
    def test_set_attributes(self):
        ob = lt.DataObject()
        ob.z = 200
        ob.a = 100
        self.assertEqual([('a', 100), ('z', 200)], sorted(ob.__dict__.items()))

        # test semi-immutability (can't overwrite existing attributes)
        with self.assertRaises(AttributeError):
            ob.a = 101

        # equality tests
        ob2 = lt.DataObject(**{'a': 100, 'z': 200})
        self.assertEqual(ob2, ob)

        ob2.b = 'blah'
        self.assertNotEqual(ob, ob2)

        del ob2.b
        self.assertEqual(ob2, ob)

        del ob2.a
        del ob2.z

        with self.assertRaises(KeyError):
            ob2['a']
        ob2['a'] = 10
        ob2['a']

        with self.assertRaises(KeyError):
            ob2['a'] = 10

        self.assertEqual("{'a': 10}", repr(ob2))

class TestTableTypes(unittest.TestCase):
    def test_types(self):
        
        # check that Table and Index are recognized as Sequence and Mapping types
        t = lt.Table()
        self.assertTrue(isinstance(t, lt.Sequence))
        
        t.create_index("x")
        self.assertTrue(isinstance(t.get_index('x'), lt.Mapping))

        # make sure get_index returns a read-only access to the underlying index
        with self.assertRaises(Exception):
            t.get_index("x")['a'] = 100


class AbstractContentTypeFactory:
    data_object_type = None

    @classmethod
    def make_data_object(cls, a, b, c):
        return cls.data_object_type(a=a, b=b, c=c)

class UsingDataObjects(AbstractContentTypeFactory):
    data_object_type = lt.DataObject

class UsingNamedtuples(AbstractContentTypeFactory):
    data_object_type = DataTuple

class UsingSlottedObjects(AbstractContentTypeFactory):
    data_object_type = Slotted

class UsingSimpleNamespace(AbstractContentTypeFactory):
    data_object_type = SimpleNamespace

if dataclasses is not None:
    class UsingDataclasses(AbstractContentTypeFactory):
        data_object_type = DataDataclass


def load_table(table, rec_factory_fn, table_size):
    test_size = table_size
    for aa in range(test_size):
        for bb in range(test_size):
            for cc in range(test_size):
                table.insert(rec_factory_fn(aa, bb, cc))


def make_test_table(rec_factory_fn, table_size):
    table = lt.Table()
    load_table(table, rec_factory_fn, table_size)
    return table


def make_dataobject_from_ob(rec):
    return lt.DataObject(**dict((k, getattr(rec, k)) for k in lt._object_attrnames(rec)))


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


class TableCreateTests_DataObjects(unittest.TestCase, TableCreateTests, UsingDataObjects):
    pass

class TableCreateTests_Namedtuples(unittest.TestCase, TableCreateTests, UsingNamedtuples):
    pass

class TableCreateTests_Slotted(unittest.TestCase, TableCreateTests, UsingSlottedObjects):
    pass

class TableCreateTests_SimpleNamespace(unittest.TestCase, TableCreateTests, UsingSimpleNamespace):
    pass

if dataclasses is not None:
    class TableCreateTests_Dataclasses(unittest.TestCase, TableCreateTests, UsingDataclasses):
        pass


class TableListTests:
    def _test_init(self):
        self.test_size = 3
        self.t1 = make_test_table(self.make_data_object, self.test_size)
        self.test_rec = self.make_data_object(1,1,1)

    def test_contains(self):
        self._test_init()
        self.assertTrue(self.test_rec in self.t1, "failed 'in' (contains) test")

    def test_index_find(self):
        self._test_init()
        self.assertEqual(13, self.t1.index(self.test_rec), "failed 'in' (contains) test")

    def test_remove(self):
        self._test_init()
        rec = self.make_data_object(1, 1, 1)
        prev_len = len(self.t1)
        self.t1.remove(rec)
        self.assertFalse(rec in self.t1, "failed to remove record from table (contains)")
        self.assertEqual(prev_len-1, len(self.t1), "failed to remove record from table (len)")

    def test_index_access(self):
        self._test_init()
        self.assertEqual(self.test_rec, self.t1[13], "failed index access test")

    def test_count(self):
        self._test_init()
        self.assertTrue(self.t1.count(self.test_rec) == 1, "failed count test")

    def test_reversed(self):
        self._test_init()
        self.assertEqual(self.make_data_object(2,2,2), next(reversed(self.t1)), "failed reversed test")

    def test_iter(self):
        self._test_init()
        self.assertTrue(self.test_rec in self.t1, "failed 'in' (contains) test")

    def test_head_and_tail(self):
        self._test_init()
        self.t1.create_index("a")
        self.t1.create_index("c")
        self.assertEqual(set(["a", "c"]), set(self.t1._indexes.keys()), "failed to create indexes")

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
                         "failed as_html with named field format")

        html_output = self.t1[:10].as_html(fields="a b c", formats={int: "{:03d}"})
        print(html_output)
        html_lines = html_output.splitlines()
        data_line = next(h for h in html_lines if "<td>" in h)
        self.assertEqual('<tbody><tr><td><div align="right">000</div></td>'
                         '<td><div align="right">000</div></td>'
                         '<td><div align="right">000</div></td></tr>',
                         data_line,
                         "failed as_html with data type format")

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
        self.assertEqual(self.test_size ** num_fields, len(self.t1), "invalid len")
        self.t1.create_index("a")

        self.t1.clear()
        self.assertEqual(0, len(self.t1), "invalid len after clear")
        self.assertEqual(1, len(self.t1.info()["indexes"]), "invalid indexes after clear")

    def test_stats(self):
        self._test_init()
        field_names = self.t1.info()["fields"]
        num_fields = len(field_names)
        t1_stats = self.t1.stats().select("name count min max mean")
        for fieldname in field_names:
            stat_rec = t1_stats.by.name[fieldname]
            self.assertEqual(lt.DataObject(name=fieldname,
                                           count=self.test_size ** num_fields,
                                           min=0,
                                           max=self.test_size - 1,
                                           mean=(self.test_size - 1) / 2),
                             stat_rec,
                             "invalid stat for {}".format(fieldname))

        t1_stats = self.t1.stats(by_field=False)
        for stat, value in (('min', 0), ('max', self.test_size - 1), ('count', self.test_size ** num_fields),):
            for fieldname in field_names:
                self.assertEqual(value, getattr(t1_stats.by.stat[stat], fieldname),
                                 "invalid {} stat for {}".format(stat, fieldname))

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
        self.assertEqual(self.test_size ** num_fields - 1, t1_stats.by.name["a"].count)

class TableListTests_DataObjects(unittest.TestCase, TableListTests, UsingDataObjects):
    pass

class TableListTests_Namedtuples(unittest.TestCase, TableListTests, UsingNamedtuples):
    pass

class TableListTests_Slotted(unittest.TestCase, TableListTests, UsingSlottedObjects):
    pass

class TableListTests_SimpleNamespace(unittest.TestCase, TableListTests, UsingSimpleNamespace):
    pass

if dataclasses is not None:
    class TableListTests_Dataclasses(unittest.TestCase, TableListTests, UsingDataclasses):
        pass


class TableJoinTests:
    def test_simple_join(self):
        test_size = 10
        t1 = make_test_table(self.make_data_object, test_size)
        t1.create_index('a')

        t2 = lt.Table()
        t2.create_index('a')
        t2.insert(lt.DataObject(a=1, d=100))

        joined = (t1.join_on('a') + t2.join_on('a'))()
        self.assertEqual(test_size * test_size, len(joined))

        joined = (t1.join_on('a') + t2)()
        self.assertEqual(test_size * test_size, len(joined))

        joined = (t1 + t2.join_on('a'))()
        self.assertEqual(test_size * test_size, len(joined))

        t1.delete_index('a')
        with self.assertRaises(ValueError):
            joined = (t1 + t2.join_on('a'))()

        with self.assertRaises(TypeError):
            # invalid join, no kwargs listing attributes to join on
            t3 = t1.join(t2, 'a,d')

        with self.assertRaises(ValueError):
            # invalid join, no such attribute 'z'
            t3 = t1.join(t2, 'a,d,z', a='a')

        t3 = t1.join(t2, 'a,d', a='a')
        self.assertEqual(test_size * test_size, len(t3))

        t4 = t1.join(t2, a='a').select('a c d', e=lambda rec: rec.a + rec.c + rec.d)
        self.assertTrue(all(rec.e == rec.a+rec.c+rec.d for rec in t4))

        # join to empty list, should return empty table
        empty_table = lt.Table()
        empty_table.create_index('a')
        t5 = (t1.join_on('a') + empty_table)()
        self.assertEqual(0, len(t5))

    def test_outer_joins(self):
        t1 = lt.Table()
        t1.csv_import(textwrap.dedent("""\
        sku,color,size,material
        001,red,XL,cotton
        002,blue,XL,cotton/poly
        003,blue,L,linen
        004,red,M,cotton
        """))

        t2 = lt.Table()
        t2.csv_import(textwrap.dedent("""\
        sku,unit_price,size
        001,10,L
        001,12,XL
        002,11,
        004,9,
        """), transforms={'size': lambda x: x or None})
        print(t1.info())

        t3 = t1.join(t2, auto_create_indexes=True, sku="sku")()
        print(t3.info())
        self.assertEqual(4, len(t3))

        t3 = t1.join(t2, auto_create_indexes=True, sku="sku", size="size")()("inner join")
        print(t3.info())
        self.assertEqual(1, len(t3))

        t3 = t1.join(t2, auto_create_indexes=True, join="right outer", sku="sku", size="size")()("right outer join")
        print(t3.info())
        self.assertEqual(5, len(t3))

        t3 = t1.join(t2, auto_create_indexes=True, join="left outer", sku="sku", size="size")()("left outer join")
        print(t3.info())
        self.assertEqual(3, len(t3))

        t3 = t1.join(t2, auto_create_indexes=True, join="full outer", sku="sku", size="size")()("full outer join")
        print(t3.info())
        self.assertEqual(19, len(t3))


class TableJoinTests_DataObjects(unittest.TestCase, TableJoinTests, UsingDataObjects):
    pass

class TableJoinTests_Namedtuples(unittest.TestCase, TableJoinTests, UsingNamedtuples):
    pass

class TableJoinTests_Slotted(unittest.TestCase, TableJoinTests, UsingSlottedObjects):
    pass

class TableJoinTests_SimpleNamespace(unittest.TestCase, TableJoinTests, UsingSimpleNamespace):
    pass

if dataclasses is not None:
    class TableJoinTests_Dataclasses(unittest.TestCase, TableJoinTests, UsingDataclasses):
        pass


class TableTransformTests:
    def test_sort(self):
        test_size = 10
        t1 = make_test_table(self.make_data_object, test_size)

        c_groups = 0
        for c_value, recs in itertools.groupby(t1, key=lambda rec: rec.c):
            c_groups += 1
            list(recs)
        self.assertEqual(test_size * test_size * test_size, c_groups)

        t1.sort('c')
        c_groups = 0
        for c_value, recs in itertools.groupby(t1, key=lambda rec: rec.c):
            c_groups += 1
            list(recs)
        self.assertEqual(test_size, c_groups)
        self.assertEqual(0, t1[0].c)

        t1.sort('c desc')
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

        self.assertEqual(t1_tuples, t2_tuples, "failed mixed ascending/descending multi-attribute sort")

    def test_unique(self):
        test_size = 10
        t1 = make_test_table(self.make_data_object, test_size)

        t2 = t1.unique()
        self.assertEqual(len(t1), len(t2))

        t3 = t1.unique(key=lambda rec: rec.c)
        self.assertEqual(test_size, len(t3))

class TableTransformTests_DataObjects(unittest.TestCase, TableTransformTests, UsingDataObjects):
    pass

class TableTransformTests_Namedtuples(unittest.TestCase, TableTransformTests, UsingNamedtuples):
    pass

class TableTransformTests_Slotted(unittest.TestCase, TableTransformTests, UsingSlottedObjects):
    pass

class TableTransformTests_SimpleNamespace(unittest.TestCase, TableTransformTests, UsingSimpleNamespace):
    pass

if dataclasses is not None:
    class TableTransformTests_Dataclasses(unittest.TestCase, TableTransformTests, UsingDataclasses):
        pass


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
            self.assertEqual(','.join(fieldnames), outlines[0])
            self.assertEqual(test_size**3+1, len(outlines))
            for ob, line in zip(t1, outlines[1:]):
                csv_vals = line.split(',')
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
            self.assertEqual(','.join(fieldnames), outlines[0])
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
            self.assertEqual(set(fieldnames), set(outlines[0].split(',')))
            self.assertEqual(1, len(outlines))

    def test_csv_import(self):
        data = csv_data
        incsv = io.StringIO(data)
        csvtable = lt.Table().csv_import(incsv, transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, csvtable)))
        self.assertEqual(sum(1 for line in data.splitlines() if line.strip())-1, len(csvtable))

        incsv = io.StringIO(data)
        row_prototype = self.make_data_object(0, 0, 0)
        csvtable2 = lt.Table().csv_import(incsv, transforms={'a': int, 'b': int, 'c': int}, row_class=type(row_prototype))[:3]
        
        print(type(t1[0]).__name__, t1[0])
        print(type(csvtable2[0]).__name__, csvtable2[0])
        self.assertEqual(type(t1[0]), type(csvtable2[0]))

    def test_csv_compressed_import(self):
        tt = lt.Table().csv_import("test/abc.csv", transforms=dict.fromkeys("abc", int))
        print("abc.csv", tt.info())

        compressed_files = ["abc.csv.zip", "abc.csv.gz"]
        if PY_3:
            compressed_files.append("abc.csv.xz")
        for name in compressed_files:
            tt2 = lt.Table().csv_import("test/" + name, transforms=dict.fromkeys("abc", int))
            print(name, tt2.info())
            self.assertEqual(tt.info(), tt2.info())
            self.assertEqual(sum(tt.all.a), sum(tt2.all.a))
            self.assertEqual(sum(tt.all.b), sum(tt2.all.b))
            self.assertEqual(sum(tt.all.c), sum(tt2.all.c))

        if PY_2:
            with self.assertRaises(Exception):
                name = "abc.csv.xz"
                tt2 = lt.Table().csv_import("test/" + name, transforms=dict.fromkeys("abc", int))

        tt2 = lt.Table().json_import("test/abc.json.gz")
        print("abc.json.gz", tt2.info())
        self.assertEqual(tt.info(), tt2.info())
        self.assertEqual(sum(tt.all.a), sum(tt2.all.a))
        self.assertEqual(sum(tt.all.b), sum(tt2.all.b))
        self.assertEqual(sum(tt.all.c), sum(tt2.all.c))

    def test_csv_filtered_import(self):
        test_size = 3
        tt = lt.Table().csv_import("test/abc.csv", transforms=dict.fromkeys("abc", int))
        print("abc.csv", tt.info())

        tt = lt.Table().csv_import("test/abc.csv", transforms=dict.fromkeys("abc", int),
                                   filters={"c": lt.Table.eq(1)})
        print(tt.info())
        self.assertEqual(test_size * test_size, len(tt))

        tt = lt.Table().csv_import("test/abc.csv", transforms=dict.fromkeys("abc", int),
                                   filters={"c": 1})
        print(tt.info())
        self.assertEqual(test_size * test_size, len(tt))

        tt = lt.Table().csv_import("test/abc.csv", transforms=dict.fromkeys("abc", int),
                                   filters={"c": lambda x: 0 < x < 2})
        print(tt.info())
        self.assertEqual(test_size * test_size, len(tt))

    def test_csv_string_import(self):
        data = csv_data
        csvtable = lt.Table().csv_import(csv_source=data, transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, csvtable)))
        self.assertEqual(sum(1 for line in data.splitlines() if line.strip())-1, len(csvtable))

        row_prototype = self.make_data_object(0, 0, 0)
        csvtable2 = lt.Table().csv_import(data, transforms={'a': int, 'b': int, 'c': int},
                                          row_class=type(row_prototype))[:3]

        print(type(t1[0]).__name__, t1[0])
        print(type(csvtable2[0]).__name__, csvtable2[0])
        self.assertEqual(type(t1[0]), type(csvtable2[0]))

    def test_csv_limit_import(self):
        data = csv_data
        import_limit = 10
        csvtable = lt.Table().csv_import(csv_source=data, transforms={'a': int, 'b': int, 'c': int},
                                         limit=import_limit)

        self.assertEqual(import_limit, len(csvtable))

        csvtable = lt.Table().csv_import(csv_source=data, transforms={'a': int, 'b': int, 'c': int},
                                         limit=0)

        self.assertEqual(0, len(csvtable))

    def test_csv_string_list_import(self):
        data = csv_data
        csvtable = lt.Table().csv_import(csv_source=data.splitlines(), transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, csvtable)))
        self.assertEqual(sum(1 for line in data.splitlines() if line.strip())-1, len(csvtable))

        row_prototype = self.make_data_object(0, 0, 0)
        csvtable2 = lt.Table().csv_import(data, transforms={'a': int, 'b': int, 'c': int},
                                          row_class=type(row_prototype))[:3]

        print(type(t1[0]).__name__, t1[0])
        print(type(csvtable2[0]).__name__, csvtable2[0])
        self.assertEqual(type(t1[0]), type(csvtable2[0]))

    def test_json_export(self):
        from itertools import permutations
        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)
        for fieldnames in permutations(list('abc')):
            out = io.StringIO()
            t1.json_export(out, fieldnames)
            out.seek(0)
            outlines = out.read().splitlines()
            out.close()

            self.assertEqual(test_size**3, len(outlines))

            for ob, line in zip(t1, outlines):
                json_dict = json.loads(line)
                t1_dataobj = make_dataobject_from_ob(ob)
                self.assertEqual(t1_dataobj, lt.DataObject(**json_dict))

    def test_json_import(self):
        data = json_data
        injson = io.StringIO(data)
        jsontable = lt.Table().json_import(injson, transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, jsontable)))
        self.assertEqual(len([d for d in data.splitlines() if d.strip()]), len(jsontable))

    def test_json_string_import(self):
        data = json_data
        jsontable = lt.Table().json_import(data, transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, jsontable)))
        self.assertEqual(len([d for d in data.splitlines() if d.strip()]), len(jsontable))

    def test_json_string_list_import(self):
        data = json_data
        jsontable = lt.Table().json_import(data.splitlines(), transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, jsontable)))
        self.assertEqual(len([d for d in data.splitlines() if d.strip()]), len(jsontable))

    def test_fixed_width_import(self):
        data = fixed_width_data
        data_file = io.StringIO(data)
        fw_spec = [('a', 0, None, int), ('b', 2, None, int), ('c', 4, None, int), ]
        tt = lt.Table().insert_many(lt.DataObject(**rec) for rec in lt.FixedWidthReader(fw_spec, data_file))

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, tt)))
        self.assertEqual(len([d for d in data.splitlines() if d.strip()]), len(tt))

    def test_fixed_width_string_import(self):
        data = fixed_width_data
        fw_spec = [('a', 0, None, int), ('b', 2, None, int), ('c', 4, None, int), ]
        tt = lt.Table().insert_many(lt.DataObject(**rec) for rec in lt.FixedWidthReader(fw_spec, data))

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, tt)))
        self.assertEqual(len([d for d in data.splitlines() if d.strip()]), len(tt))

    def test_fixed_width_string_list_import(self):
        data = fixed_width_data
        fw_spec = [('a', 0, None, int), ('b', 2, None, int), ('c', 4, None, int),]
        tt = lt.Table().insert_many(lt.DataObject(**rec) for rec in lt.FixedWidthReader(fw_spec, data.splitlines()))

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, tt)))
        self.assertEqual(len([d for d in data.splitlines() if d.strip()]), len(tt))

class TableImportExportTests_DataObjects(unittest.TestCase, TableImportExportTests, UsingDataObjects):
    pass

class TableImportExportTests_Namedtuples(unittest.TestCase, TableImportExportTests, UsingNamedtuples):
    pass

class TableImportExportTests_Slotted(unittest.TestCase, TableImportExportTests, UsingSlottedObjects):
    pass

class TableImportExportTests_SimpleNamespace(unittest.TestCase, TableImportExportTests, UsingSimpleNamespace):
    pass

if dataclasses is not None:
    class TableImportExportTests_Dataclasses(unittest.TestCase, TableImportExportTests, UsingDataclasses):
        pass


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

class TablePivotTests_DataObjects(unittest.TestCase, TablePivotTests, UsingDataObjects):
    pass

class TablePivotTests_Namedtuples(unittest.TestCase, TablePivotTests, UsingNamedtuples):
    pass

class TablePivotTests_Slotted(unittest.TestCase, TablePivotTests, UsingSlottedObjects):
    pass

class TablePivotTests_SimpleNamespace(unittest.TestCase, TablePivotTests, UsingSimpleNamespace):
    pass

if dataclasses is not None:
    class TablePivotTests_Dataclasses(unittest.TestCase, TablePivotTests, UsingDataclasses):
        pass

class InitialTest(unittest.TestCase):
    if sys.version_info[:2] <= (2, 6):
        print('unit_tests.py only runs on Python 2.7 or later')
        sys.exit(0)

    from littletable import (
        __version__ as littletable_version,
        __versionTime__ as littletable_version_time,
        __version_info__ as littletable_version_info,
    )

    print(
        "Beginning test of littletable, version",
        littletable_version,
        littletable_version_time,
    )
    print(littletable_version_info)
    print("Python version", sys.version)
    print()


if __name__ == '__main__':

    unittest.main()
