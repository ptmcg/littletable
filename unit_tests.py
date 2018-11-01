import unittest
import littletable as lt
import itertools
import json
from collections import namedtuple
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
        self.assertEqual(sorted(ob.__dict__.items()), [('a', 100), ('z', 200)])

        # test semi-immutability (can't overwrite existing attributes)
        with self.assertRaises(AttributeError):
            ob.a = 101

        # equality tests
        ob2 = lt.DataObject(**{'a': 100, 'z': 200})
        self.assertEqual(ob, ob2)

        ob2.b = 'blah'
        self.assertNotEqual(ob, ob2)

        del ob2.b
        self.assertEqual(ob, ob2)

        del ob2.a
        del ob2.z

        with self.assertRaises(KeyError):
            ob2['a']
        ob2['a'] = 10
        ob2['a']

        with self.assertRaises(KeyError):
            ob2['a'] = 10

        self.assertEqual(repr(ob2), "{'a': 10}")

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


class UsingDataObjects:
    @staticmethod
    def make_data_object(a, b, c):
        return lt.DataObject(a=a, b=b, c=c)

class UsingNamedtuples:
    @staticmethod
    def make_data_object(a, b, c):
        return DataTuple(a=a, b=b, c=c)

class UsingSlottedObjects:
    @staticmethod
    def make_data_object(a, b, c):
        return Slotted(a=a, b=b, c=c)

class UsingSimpleNamespace:
    @staticmethod
    def make_data_object(a, b, c):
        return SimpleNamespace(a=a, b=b, c=c)
    

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
        self.assertEqual(table.by.a[4], self.make_data_object(4, 5, 6))

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

        self.assertEqual(len(table.where(a=5)), test_size*test_size)
        self.assertEqual(len(table.where(a=-1)), 0)

    def test_where_equals_none(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)

        self.assertEqual(len(table.where(a=5, b=test_size)), 0)

    def test_where_equals_with_index(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)
        table.create_index('a')

        self.assertEqual(len(table.where(a=5)), test_size*test_size)
        self.assertEqual(len(table.where(a=-1)), 0)

    def test_where_range(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)

        self.assertEqual(len(table.where(lambda rec: rec.a == rec.b)), test_size*test_size)

    def test_get_slice(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)

        subtable = table[0::test_size]
        self.assertEqual(len(subtable), test_size * test_size)

    def test_indexing(self):
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        make_rec = lambda aa, bb, cc: self.make_data_object(chars[aa], chars[bb], chars[cc])
        test_size = 10
        table = make_test_table(make_rec, test_size)
        table.create_index('a')

        self.assertTrue('A' in table.by.a)
        self.assertTrue('AA' not in table.by.a)
        self.assertEqual(len(table.by.a['B']), test_size * test_size)
        self.assertTrue(isinstance(table.by.a['B'], lt.Table))
        with self.assertRaises(AttributeError):
            table.by.z

        self.assertEqual(len(table.by.a.keys()), test_size)

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

        # now drop index and recreate not permitting None, should raise exception
        table.delete_index('a')
        with self.assertRaises(KeyError):
            table.create_index('a', unique=True, accept_none=False)

        table.create_index('a', unique=True, accept_none=True)
        table.create_index('c')

        import pprint
        info = table.info()
        pprint.pprint(info)
        self.assertEqual(info['name'], 'Table_1')
        self.assertEqual(list(sorted(info['fields'])), list('abc'))
        self.assertEqual(list(sorted(info['indexes'])), [('a', True), ('c', False)])
        self.assertEqual(info['len'], 1001)

    def test_delete_by_filter(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)

        self.assertEqual(table.delete(b=5), test_size*test_size)
        self.assertEqual(len(table), test_size*test_size*(test_size-1))
        self.assertEqual(table.delete(b=-1), 0)

        self.assertEqual(table.delete(), 0)

    def test_remove_many(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)

        table.remove_many(table.where(lambda t: t.a % 2))
        self.assertEqual(len(table), test_size*test_size*test_size/2)
        table_len = len(table)
        table.remove(table[1])
        self.assertEqual(len(table), table_len-1)

    def test_add_new_field(self):
        test_size = 10
        table = make_test_table(self.make_data_object, test_size)

        # only DataObjects are mutable in these tests
        if isinstance(table[0], lt.DataObject):
            table.add_field('d', lambda rec: rec.a+rec.b+rec.c)

            table.create_index('d')
            self.assertEqual(len(table.by.d.keys()), len(range(0, 27+1)))

    def test_add_two_tables(self):
        test_size = 10
        t1 = make_test_table(self.make_data_object, test_size)
        make_rec = lambda a,b,c: self.make_data_object(a+test_size, b, c)
        t2 = make_test_table(make_rec, test_size)

        self.assertEqual(len(t1+t2), test_size*test_size*test_size*2)
        self.assertEqual(len(t1), test_size * test_size * test_size)

        t1 += t2
        self.assertEqual(len(t1), test_size * test_size * test_size * 2)

        offset = test_size * test_size
        t3 = t1 + (self.make_data_object(rec.a+offset, rec.b, rec.c) for rec in t2)
        self.assertEqual(len(t3), test_size * test_size * test_size * 3)

class TableCreateTests_DataObjects(unittest.TestCase, TableCreateTests, UsingDataObjects):
    pass

class TableCreateTests_Namedtuples(unittest.TestCase, TableCreateTests, UsingNamedtuples):
    pass

class TableCreateTests_Slotted(unittest.TestCase, TableCreateTests, UsingSlottedObjects):
    pass

class TableCreateTests_SimpleNamespace(unittest.TestCase, TableCreateTests, UsingSimpleNamespace):
    pass


class TableListTests:
    def _test_init(self):
        test_size = 3
        self.t1 = make_test_table(self.make_data_object, test_size)
        self.test_rec = self.make_data_object(1,1,1)

    def test_contains(self):
        self._test_init()
        self.assertTrue(self.test_rec in self.t1, "failed 'in' (contains) test")

    def test_index_find(self):
        self._test_init()
        self.assertEqual(self.t1.index(self.test_rec), 13, "failed 'in' (contains) test")

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

class TableListTests_DataObjects(unittest.TestCase, TableListTests, UsingDataObjects):
    pass

class TableListTests_Namedtuples(unittest.TestCase, TableListTests, UsingNamedtuples):
    pass

class TableListTests_Slotted(unittest.TestCase, TableListTests, UsingSlottedObjects):
    pass

class TableListTests_SimpleNamespace(unittest.TestCase, TableListTests, UsingSimpleNamespace):
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
        self.assertEqual(len(joined), test_size * test_size)

        joined = (t1.join_on('a') + t2)()
        self.assertEqual(len(joined), test_size * test_size)

        joined = (t1 + t2.join_on('a'))()
        self.assertEqual(len(joined), test_size * test_size)

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
        self.assertEqual(len(t3), test_size * test_size)

        t4 = t1.join(t2, a='a').select('a c d', e=lambda rec: rec.a + rec.c + rec.d)
        self.assertTrue(all(rec.e == rec.a+rec.c+rec.d for rec in t4))

        # join to empty list, should return empty table
        empty_table = lt.Table()
        empty_table.create_index('a')
        t5 = (t1.join_on('a') + empty_table)()
        self.assertEqual(len(t5), 0)

class TableJoinTests_DataObjects(unittest.TestCase, TableJoinTests, UsingDataObjects):
    pass

class TableJoinTests_Namedtuples(unittest.TestCase, TableJoinTests, UsingNamedtuples):
    pass

class TableJoinTests_Slotted(unittest.TestCase, TableJoinTests, UsingSlottedObjects):
    pass

class TableJoinTests_SimpleNamespace(unittest.TestCase, TableJoinTests, UsingSimpleNamespace):
    pass


class TableTransformTests:
    def test_sort(self):
        test_size = 10
        t1 = make_test_table(self.make_data_object, test_size)

        c_groups = 0
        for c_value, recs in itertools.groupby(t1, key=lambda rec: rec.c):
            c_groups += 1
            list(recs)
        self.assertEqual(c_groups, test_size * test_size * test_size)

        t1.sort('c')
        c_groups = 0
        for c_value, recs in itertools.groupby(t1, key=lambda rec: rec.c):
            c_groups += 1
            list(recs)
        self.assertEqual(c_groups, test_size)
        self.assertEqual(t1[0].c, 0)

        t1.sort('c desc')
        self.assertEqual(t1[0].c, test_size-1)

    def test_unique(self):
        test_size = 10
        t1 = make_test_table(self.make_data_object, test_size)

        t2 = t1.unique()
        self.assertEqual(len(t2), len(t1))

        t3 = t1.unique(key=lambda rec: rec.c)
        self.assertEqual(len(t3), test_size)

class TableTransformTests_DataObjects(unittest.TestCase, TableTransformTests, UsingDataObjects):
    pass

class TableTransformTests_Namedtuples(unittest.TestCase, TableTransformTests, UsingNamedtuples):
    pass

class TableTransformTests_Slotted(unittest.TestCase, TableTransformTests, UsingSlottedObjects):
    pass

class TableTransformTests_SimpleNamespace(unittest.TestCase, TableTransformTests, UsingSimpleNamespace):
    pass


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
            self.assertEqual(len(outlines), test_size**3+1)
            for ob, line in zip(t1, outlines[1:]):
                csv_vals = line.split(',')
                self.assertTrue(all(
                    int(csv_vals[i]) == getattr(ob, fld) for i, fld in enumerate(fieldnames)
                ))

    def test_csv_import(self):
        data = """\
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
2,2,2"""
        incsv = io.StringIO(data)
        csvtable = lt.Table().csv_import(incsv, transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, csvtable)))
        self.assertEqual(len(data.splitlines())-1, len(csvtable))

        incsv = io.StringIO(data)
        row_prototype = self.make_data_object(0, 0, 0)
        csvtable2 = lt.Table().csv_import(incsv, transforms={'a': int, 'b': int, 'c': int}, row_class=type(row_prototype))[:3]
        
        print(type(t1[0]).__name__, t1[0])
        print(type(csvtable2[0]).__name__, csvtable2[0])
        self.assertEqual(type(t1[0]), type(csvtable2[0]))

    def test_csv_string_import(self):
        data = """\
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
2,2,2"""
        csvtable = lt.Table().csv_import(csv_source=data, transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, csvtable)))
        self.assertEqual(len(data.splitlines()) - 1, len(csvtable))

        row_prototype = self.make_data_object(0, 0, 0)
        csvtable2 = lt.Table().csv_import(data, transforms={'a': int, 'b': int, 'c': int},
                                          row_class=type(row_prototype))[:3]

        print(type(t1[0]).__name__, t1[0])
        print(type(csvtable2[0]).__name__, csvtable2[0])
        self.assertEqual(type(t1[0]), type(csvtable2[0]))

    def test_csv_string_list_import(self):
        data = """\
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
2,2,2"""
        csvtable = lt.Table().csv_import(csv_source=data.splitlines(), transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, csvtable)))
        self.assertEqual(len(data.splitlines()) - 1, len(csvtable))

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

            self.assertEqual(len(outlines), test_size**3)

            for ob, line in zip(t1, outlines):
                json_dict = json.loads(line)
                t1_dataobj = make_dataobject_from_ob(ob)
                self.assertEqual(lt.DataObject(**json_dict), t1_dataobj)

    def test_json_import(self):
        data = """\
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
            {"a": 2, "b": 2, "c": 2}"""
        injson = io.StringIO(data)
        jsontable = lt.Table().json_import(injson, transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, jsontable)))
        self.assertEqual(len(data.splitlines()), len(jsontable))

    def test_json_string_import(self):
        data = """\
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
            {"a": 2, "b": 2, "c": 2}"""
        jsontable = lt.Table().json_import(data, transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, jsontable)))
        self.assertEqual(len(data.splitlines()), len(jsontable))

    def test_json_string_list_import(self):
        data = """\
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
            {"a": 2, "b": 2, "c": 2}"""
        jsontable = lt.Table().json_import(data.splitlines(), transforms={'a': int, 'b': int, 'c': int})

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, jsontable)))
        self.assertEqual(len(data.splitlines()), len(jsontable))

    def test_fixed_width_import(self):
        data = """\
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
2 2 2"""
        data_file = io.StringIO(data)
        fw_spec = [('a', 0, None, int), ('b', 2, None, int), ('c', 4, None, int), ]
        tt = lt.Table().insert_many(lt.DataObject(**rec) for rec in lt.FixedWidthReader(fw_spec, data_file))

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, tt)))
        self.assertEqual(len(data.splitlines()), len(tt))

    def test_fixed_width_string_import(self):
        data = """\
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
        fw_spec = [('a', 0, None, int), ('b', 2, None, int), ('c', 4, None, int), ]
        tt = lt.Table().insert_many(lt.DataObject(**rec) for rec in lt.FixedWidthReader(fw_spec, data))

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, tt)))
        self.assertEqual(len(list(filter(None, data.splitlines()))), len(tt))

    def test_fixed_width_string_list_import(self):
        data = """\
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
2 2 2"""
        fw_spec = [('a', 0, None, int), ('b', 2, None, int), ('c', 4, None, int),]
        tt = lt.Table().insert_many(lt.DataObject(**rec) for rec in lt.FixedWidthReader(fw_spec, data.splitlines()))

        test_size = 3
        t1 = make_test_table(self.make_data_object, test_size)

        self.assertTrue(all(make_dataobject_from_ob(rec1) == rec2 for rec1, rec2 in zip(t1, tt)))
        self.assertEqual(len(data.splitlines()), len(tt))

class TableImportExportTests_DataObjects(unittest.TestCase, TableImportExportTests, UsingDataObjects):
    pass

class TableImportExportTests_Namedtuples(unittest.TestCase, TableImportExportTests, UsingNamedtuples):
    pass

class TableImportExportTests_Slotted(unittest.TestCase, TableImportExportTests, UsingSlottedObjects):
    pass

class TableImportExportTests_SimpleNamespace(unittest.TestCase, TableImportExportTests, UsingSimpleNamespace):
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


if __name__ == '__main__':
    if sys.version_info[:2] == (2, 6):
        print('unit_tests.py only runs on Python 2.7 or later')
        sys.exit(0)

    unittest.main()
