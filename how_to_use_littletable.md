How to Use littletable
======================

  * [Introduction](#introduction)
  * [Creating a table](#creating-a-table)
  * [Inserting objects](#inserting-objects)
  * [Importing data from CSV files](#importing-data-from-csv-files)
  * [Import/export data to Excel files](#importexport-to-excel-files-xlsx)
  * [Tabular output](#tabular-output)
  * [types.SimpleNamespace](#typessimplenamespace)
  * [Removing objects](#removing-objects)
  * [Indexing attributes](#indexing-attributes)
  * [Querying with indexed attributes](#querying-with-indexed-attributes)
  * [Querying for exact matching attribute values](#querying-for-exact-matching-attribute-values)
  * [Querying for attribute value ranges](#querying-for-attribute-value-ranges)
  * [Splitting a table using a criteria function](#splitting-a-table-using-a-criteria-function)
  * [Full-text search on text attributes](#full-text-search-on-text-attributes)
  * [Simple statistics on Tables of numeric values](#simple-statistics-on-tables-of-numeric-values)
  * [Importing data from fixed-width text files](#importing-data-from-fixed-width-text-files)
  * [Joining tables](#joining-tables)
  * [Pivoting a table](#pivoting-a-table)
  * [littletable and pandas](#littletable-and-pandas)
  * [littletable and SQLite](#littletable-and-sqlite)
  * [Some simple littletable recipes](#some-simple-littletable-recipes)


Introduction
------------

`littletable` is a simple Python module to make it easy to work with collections 
of objects as if they were records in a database table.  `littletable` augments 
normal Python list access with:
- indexing by key attributes
- joining multiple tables by common attribute values
- querying for matching objects by one or more attributes
- data pivoting on 1 or more attributes

It is not necessary to define a table schema for tables in `littletable`; the 
schema of the data emerges from the attributes of the stored objects, and those 
used to define indexes and queries.

Indexes can be created and dropped at any time. An index can be defined to have 
unique or non-unique key values, and whether or not to allow null values.

Tables can be persisted to and from CSV files using `csv_export()` and `csv_import()`.

Instead of returning DataSets or rows of structured values, `littletable` queries 
return new Tables. This makes it easy to arrive at a complex query by a sequence 
of smaller steps.  The resulting values are also easily saved to a CSV file, 
like any other `littletable` table.


Creating a table
----------------
Creating a table is simple, just create an instance of `Table`:
```python
t = Table()
```
    
If you want, you can name the table at creation time, or any time later. 
```python
t = Table("customers")
```
    
or

```python
t = Table()
t("customers")
```

Table names are not necessary for queries or updates, as they would be in SQL.  
Table names can be useful in diagnosing problems, as they will be included in 
exception messages. Table joins also use the names of the source tables to 
create a helpful name for the resulting data table.

Once you have created the Table, you can then use `insert` or `insert_many` to
populate the table, or use one of the `import` methods (such as `csv_import`,
`tsv_import`, or `json_import`) to load the table from an external file or data
string.


Inserting objects
-----------------
From within your Python code, you can create objects and add them to the table using
`insert()` and `insert_many()`. Any object can be inserted into a table, using:

```python
t.insert(obj)
t.insert_many(objlist)
```

Performance tip: Calling `insert_many()` with a list of objects will perform better than calling
`insert()` in a loop.

`littletable` supports records that are user-defined types (including those defined
using `__slots__`), `dataclasses`, `namedtuple`s, and `SimpleNamespace`s. Python objects
defined using `attrs`, `pydantic`, and `traits/traitlets` packages are also supported.
Python `dict`s can be used; they will be stored as `SimpleNamespace`s so that the `dict` fields
will be accessible as object attributes.


Importing data from CSV files
-----------------------------
You can easily import a CSV file into a `Table` using `Table.csv_import()`:

```python
t = Table().csv_import("my_data.csv")
```

In place of a local file name, you can also specify an HTTP url:

```python
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
iris_table = Table('iris').csv_import(url, fieldnames=names)
```

You can also directly import CSV data as a string:
```python
catalog_data = """\
sku,description,unitofmeas,unitprice
BRDSD-001,Bird seed,LB,3
BBS-001,Steel BB's,LB,5
MGNT-001,Magnet,EA,8"""

catalog = Table("catalog")
catalog.create_index("sku", unique=True)
catalog.csv_import(catalog_data, transforms={'unitprice': int})
```

If you are working with a very large CSV file and just trying to see 
what the structure is, add `limit=100` to only read the first 100 rows.

You can also pre-screen data as it is read from the input file by passing
a `filters={attr: filter_fn, ...}` argument. Each filter function is called
on the newly-read object _before_ it is added to the table. `filter_fn` 
can be any function that takes a single argument of the type of the given
attribute, and returns `True` or `False`. If `False`, the record does not get
added to the table.

```python
# import only the first 100 items that match the filter
# (product_category == "Home and Garden")
catalog.csv_import(catalog_data, 
                   filters={"product_category": Table.eq("Home and Garden")},
                   limit=100)
```

Since CSV files do not keep any type information, `littletable` will use the
`SimpleNamespace` type for imported records. You can specify your own type by 
passing `row_class=MyType` to `csv_import`. The type must be initializable
using the form `MyType(**attributes_dict)`. `namedtuples` and `SimpleNamespace`
both support this form.

Performance tip: For very large files, it is faster to load data using
a `dataclass` or `namedtuple` than to use the default `SimpleNamespace` class.
Get the fields using
a 10-item import using `limit=10`, and then define the `namedtuple` using the 
fields from `table.info()["fields"]`.

Files containing JSON-formatted records can be similarly imported using 
`Table.json_import()`, and tab-separated files can be imported using
`Table.tsv_import()`. 

`littletable` can also read CSV, TSV, etc. content directly from a simple .zip,
.gz, or .xz archive, assuming that the file name of the compressed file is the 
same as the original file with ".zip" or ".gz" or ".xz" added.

Note: if you find you cannot import .xz or .lzma files, getting the Python error
`ModuleNotFoundError :_lzma`, you can remedy this by rebuilding Python after 
installing the `lzma-dev` library. On Ubuntu for example, this is done using:

    $ sudo apt-get install liblzma-dev

Then rebuild and reinstall Python.

Import/export to Excel files (.xlsx)
------------------------------------
`littletable` can read and write local Excel spreadsheet files:

```python
tbl = lt.Table().excel_import("data_table.xlsx")
tbl.excel_export("new_table.xlsx")
```

Data values from Excel get converted to standard Python types where possible.
A spreadsheet containing the following data:

| name | value | type |
|---|---|---|
| a | 100 | int |
| b | 3.14159 | float |
| c | None | null |
| d | 2021-12-25 00:00:00 | date |
| e | Floyd | str |
| f |   | space |
| g | 𝚃𝖞𝐩𝓮𝖤𝔯𝘳º𝗿 | str |
| h | True | bool |
| i | =TODAY() | formula |
| j | 0 | None |
| k | None | None |

Can be imported and the data values will be automatically
converted as shown below:

```python
xl = lt.Table().excel_import("../test/data_types.xlsx")

for row in xl:
    print(row.name, repr(row.value), type(row.value), row.type)


a 100 <class 'int'> int
b 3.14159 <class 'float'> float
c None <class 'NoneType'> null
d datetime.datetime(2021, 12, 25, 0, 0) <class 'datetime.datetime'> date
e 'Floyd' <class 'str'> str
f ' ' <class 'str'> space
g '𝚃𝖞𝐩𝓮𝖤𝔯𝘳º𝗿' <class 'str'> str
h True <class 'bool'> bool
i '=TODAY()' <class 'str'> formula
j 0 <class 'int'> None
k None <class 'NoneType'> None
```

Tabular output
--------------
To produce a nice tabular output for a table, you can use the embedded support for
the `rich` module, `as_html()` in Jupyter Notebook, or the `tabulate` module:

- Using `table.present()` (implemented using `rich`; `present()` accepts `rich` Table 
  keyword args):
```python
table(title_str).present(fields=["col1", "col2", "col3"])
```

  or

```python
# use select() to limit the columns to be shown
table.select("col1 col2 col3")(title_str).present(caption="caption text")
```

- Using `Jupyter Notebook`:

```python
from IPython.display import HTML, display
display(HTML(table.as_html()))
```

- Using `tabulate`:
```python
# use map(vars, table) to get each table record as a dict, then pass to tabulate with
# headers="keys" to auto-define headers
print(tabulate(map(vars, table), headers="keys"))
```

- Output as Markdown

```python
print(table.as_markdown())
```

You can display groups in your tables by specifying a particular field on which to group.
Pass the groupby argument to present(), as_html() or as_markdown() with the name of the
field, and consecutive duplicate values for that field will be suppressed.


types.SimpleNamespace
---------------------
If your program does not have a type for inserting into your table, you can use 
`types.SimpleNamespace`:

```python
from types import SimpleNamespace
bob = {"name": "Bob", "age": 19}
t.insert(SimpleNamespace(**bob))
```

Or just use `dict`s directly (which `littletable` will convert to `SimpleNamespace`s)

```python
bob = {"name": "Bob", "age": 19}
t.insert(bob)
```

_(`DataObjects` are a legacy type from Python 2.6 - Python 3, before the availability of
`types.SimpleNamespace`. The `DataObject` class will be deprecated
in a future release.)_


Removing objects
----------------
Objects can be removed individually or by passing a list (or `Table`) of
objects:

```python
t.remove(obj)
t.remove_many(objlist)
t.remove_many(t.where(a=100))
```

They can also be removed by numeric index or slice using the Python `del` statement, just
like removing from a list:

```python
del t[0]
del t[-1]
del t[3:10]
```

Finally, items can be removed using `Table.pop()`, which like `list.pop` defaults to removing
the last item in the table, and returns the removed item:

```python
obj = t.pop(12)
obj = t.pop()
```

Indexing attributes
-------------------
Use `create_index` to add an index to a `Table`. Indexes can be unique or
non-unique. If the table is not empty and the index to be created is
`unique=True`, the uniqueness of the index attribute across the existing
records is verified before creating the index, raising `KeyError` and
listing the duplicated value.

If a unique index is created, then retrieving using that index will
return the single matching object, or raise `KeyError`.

If a non-unique index is created, a `Table` is returned of all the matching
objects. If no objects match, an empty `Table` is returned.

```python
employees.create_index('employee_id', unique=True)
employees.create_index('zipcode')

# unique indexes return a single object
print(employees.by.employee_id["D1729"].name)

# non unique indexes return a new Table
for emp in employees.by.zipcode["12345"]:
    print(e.name)
```


Querying with indexed attributes
--------------------------------

If accessing a table using a unique index, giving a key value will 
return the single matching record, or raise `KeyError`.

```python
employees.by.employee_id['00086']
employees.by.employee_id['invalid_id']
#    raises KeyError: "no such value 'invalid_id' in index 'employee_id'"
```

If accessing a table using a non-unique index, will return a new `Table` 
containing all matching records. If there are no matching records, the 
returned table will be empty.

```python
employees.by.state['CA']
employees.by.dept['Sales']
employees.by.dept['Salex']  # no such department
#    returns empty table
```


Querying for exact matching attribute values
--------------------------------------------
Calling `Table.where()` with named attributes will return a Table of
all records matching all the arguments:

```python
employees.where(zipcode="12345", title="Manager")
student.where(**{"class":"Algebra"})
```

It is not necessary for the attributes to be indexed to use `Table.where()`.


Querying for attribute value ranges
-----------------------------------
`Table.where()` supports performing queries on one or more exact
matches against entries in the table:

```python
employees.where(dept="Engineering")
```

`Table.where()` will also accept a callable that takes a record and
returns a bool to indicate if the record is a match:

```python
employees.where(lambda emp: emp.salary > 50000)
```

`littletable` also includes _comparators_ to make range-checking easier to
write. The following table lists the comparators, plus examples of their
usage:

|Comparator|Example|Comparison performed|
|---|:---|:---|
| `lt`           |  `attr=Table.lt(100)`           |  `attr < 100` |
| `le`           |  `attr=Table.le(100)`           |  `attr <= 100` |
| `gt`           |  `attr=Table.gt(100)`           |  `attr > 100` |
| `ge`           |  `attr=Table.ge(100)`           |  `attr >= 100` |
| `eq`           |  `attr=Table.eq(100)`           |  `attr == 100` |
| `ne`           |  `attr=Table.ne(100)`           |  `attr != 100` |
| `is_none`      |  `attr=Table.is_none())`        |  `attr is None` |
| `is_not_none`  |  `attr=Table.is_not_none())`    |  `attr is not None` |
| `is_null`      |  `attr=Table.is_null())`        |  `attr is None, "", or omitted` |
| `is_not_null`  |  `attr=Table.is_not_null())`    |  `attr is not None or ""` |
| `startswith`   |  `attr=Table.startswith("ABC")` |  `attr.startswith("ABC")` |
| `endswith`     |  `attr=Table.endswith("XYZ")`   |  `attr.endswith("XYZ")` |
| `re_match`     |  `attr=Table.re_match(r".*%.*")` | `re.match(r".*%.*", attr)` |
| `between`      |  `attr=Table.between(100, 200)`  | `100 < attr < 200` |
| `within`       |  `attr=Table.within(100, 200)`   | `100 <= attr <= 200` |
| `in_range`     |  `attr=Table.in_range(100, 200)` | `100 <= attr < 200` |
| `is_in`        |  `attr=Table.is_in((1, 2, 3))`  |  `attr in (1,2,3)` |
| `not_in`       |  `attr=Table.not_in((1, 2, 3))` |  `attr not in (1,2,3)` |

More examples of comparators in actual Python code:

```python
employees.where(salary=Table.gt(50000))
employees.where(dept=Table.is_in(["Sales", "Marketing"]))

jan_01 = date(2000, 1, 1)
mar_31 = date(2000, 3, 31)
apr_01 = date(2000, 4, 1)

first_qtr_sales = sales.where(date=Table.within(jan_01, mar_31))
first_qtr_sales = sales.where(date=Table.in_range(jan_01, apr_01))

# get customers whose address includes an apartment number
has_apt = customers.where(address_apt_no=Table.is_not_null())

# get employees whose first name starts with "X"
x_names = employees.where(name=Table.startswith("X"))

# get log records that match a regex (any word starts with 
# "warn" in the log description)
# (re_match will accept re flags argument)
warnings = log.where(description = Table.re_match(r".*\bwarn", flags=re.I)
```

Comparators can also be used as filter functions for import methods.


Splitting a table using a criteria function
-------------------------------------------
You can divide a `littletable.Table` into 2 new tables using
`Table.splitby`.  `Table.splitby` takes a predicate function that takes 
a table record and returns True or False, and returns two tables:
a table with all the rows that returned False and a table with all the 
rows that returned True. Will also accept a string indicating a particular
field name, and uses `bool(getattr(rec, field_name))` for the predicate
function.

```python
# split on records based on even/odd of a value attribute
is_odd = lambda x: bool(x % 2)
evens, odds = tbl.splitby(lambda rec: is_odd(rec.value))

# split on an integer field: 0 will be treated as False, >0 as True
has_no_cars, has_cars = tbl.splitby("number_of_cars_owned")

# split on a field that may be None or ""
nulls, not_nulls = tbl.splitby("optional_data_field")
```


Full-text search on text attributes
-----------------------------------
`littletable` can perform a rudimentary version of full-text search against
attributes composed of multiple-word contents (such as item descriptions, 
comments, etc.). To perform a full-text search, a search index must first be
created, using `Table.create_search_index()`, naming the attribute to be 
indexed, and optionally any stop words that should be ignored.

Afterward, queries can be run using `table.search.attribute(query)`, 
where `attribute` is the attribute that was indexed, and `query` is
a list or space-delimited string of search terms. Search terms may be
prefixed by '++' or '--' to indicate required or prohibited terms, or
'+' or '-' for preferred or non-preferred terms. The search function
uses these prefixes to compute a matching score, and the matching records
are returned in descending score order, along with their scores, and optionally
each record's parsed keywords.

In addition to the query, you may also specify a limit, and whether to include
each entry's indexed search words.

Example:

```python
recipe_data = textwrap.dedent("""\
    title,ingredients
    Tuna casserole,tuna noodles cream of mushroom soup
    Hawaiian pizza,pizza dough pineapple ham tomato sauce
    BLT,bread bacon lettuce tomato mayonnaise
    Bacon cheeseburger,ground beef bun lettuce ketchup mustard pickle cheese bacon
    """)
recipes = lt.Table().csv_import(recipe_data)

recipes.create_search_index("ingredients")
matches = recipes.search.ingredients("+bacon tomato --pineapple")
```

Search indexes will become invalid if records are added or removed from the table 
after the index has been created. If they are not rebuilt, subsequent searches
will raise the `SearchIndexInconsistentError` exception.


Simple statistics on Tables of numeric values
---------------------------------------------
`Table.stats()` will perform simple mean, variance, and standard deviation
calculations by attribute on records in a table. The results are returned
in a new `Table` that can be keyed by attribute (with "mean", "variance", etc.
attributes), or by statistic (keyed by "mean", etc., with attributes matching
those in the source `Table`). Non-numeric values are implicitly omitted from
the statistics calculations.

```python
import littletable as lt

t1 = lt.Table()
t1.csv_import("""\
a,b,c
100,101,102
110,220,99
108,130,109""", transforms=dict(a=int, b=int, c=int))

t1_stats = t1.stats()
t1_stats.present(box=lt.box.ASCII)
print(t1_stats.by.name["a"].mean)

#    +-----------------------------------------------------------------------------+
#    | Name |        Mean | Min | Max |   Variance |     Std_Dev | Count | Missing |
#    |------+-------------+-----+-----+------------+-------------+-------+---------|
#    |  a   |       106.0 | 100 | 110 |         28 | 5.29150262… |     3 |       0 |
#    |  b   | 150.333333… | 101 | 220 | 3850.3333… | 62.0510542… |     3 |       0 |
#    |  c   | 103.333333… |  99 | 109 | 26.333333… | 5.13160143… |     3 |       0 |
#    +-----------------------------------------------------------------------------+
#    106.0

t1_stats = t1.stats(by_field=False)
t1_stats.present(box=lt.box.ASCII)
print(t1_stats.by.stat["mean"].a)

#    +------------------------------------------------------------------------+
#    | Stat     |                 A |                  B |                  C |
#    |----------+-------------------+--------------------+--------------------|
#    | mean     |             106.0 | 150.33333333333334 | 103.33333333333333 |
#    | min      |               100 |                101 |                 99 |
#    | max      |               110 |                220 |                109 |
#    | variance |                28 | 3850.3333333333335 | 26.333333333333332 |
#    | std_dev  | 5.291502622129181 | 62.051054248363364 |  5.131601439446884 |
#    | count    |                 3 |                  3 |                  3 |
#    | missing  |                 0 |                  0 |                  0 |
#    +------------------------------------------------------------------------+
#    106.0
```


Importing data from fixed-width text files
----------------------------------------------
Some files contain fixed-width columns, you can use the `FixedWidthReader`
class to import these into  `littletable` `Table`s.

For data in this data file (not including the leading rows showing
column numbers):

              1         2         3         4         5         6
    0123456789012345678901234567890123456789012345678901234567890123456789
    ---
    0010GEORGE JETSON    12345 SPACESHIP ST   HOUSTON       TX 4.9
    0020WILE E COYOTE    312 ACME BLVD        TUCSON        AZ 7.3
    0030FRED FLINTSTONE  246 GRANITE LANE     BEDROCK       CA 2.6
    0040JONNY QUEST      31416 SCIENCE AVE    PALO ALTO     CA 8.1

Define the columns to import as:

```python
columns = [
    ("id_no", 0, ),
    ("name", 4, ),
    ("address", 21, ),
    ("city", 42, ),
    ("state", 56, 58, ),
    ("tech_skill_score", 59, None, float),
    ]
```

And use a `FixedWidthReader` to read the file and pass a list of 
dicts to a `Table.insert_many`:

```python
characters = lt.Table()
reader = lt.FixedWidthReader(columns, "cartoon_characters.txt")
characters.insert_many(lt.DataObject(**rec)
                       for rec in reader)
```

For each column, define:
- the attribute name
- the starting column
- (optional) the ending column (the start of the next column is 
  the default)
- (optional) a function to transform the input string (in this 
  example, 'tech_skill_score' gets converted to a float); if no 
  function is specified, str.strip() is used


Joining tables
--------------
Joining tables is one of the basic functions of relational databases. 
To join two tables, you must specify:
- the left source table and join attribute
- the right source table and join attribute
- whether the join should be performed as an inner join, left outer
  join, right outer join, or full outer join (default is an inner join)
- optionally, a list of the attributes to include in the resulting join 
  data (returned in `littletable` as a new `Table`)

`littletable` provides two different coding styles for joining tables.  
The first uses conventional object notation, with the `table.join()` 
method:

```python
customers.join(orders, custid="custid")
```

creates an inner join between the table of customers and the table of 
their respective orders, joining on both tables' `custid` attributes.

More than 2 tables can be joined in succession, since the result of a 
join is itself a `Table`:

```python
customers.join(orders, custid="custid").join(orderitems, orderid="orderid")
```
    
In this case a third table has been added, to include the actual items
that comprise each customer's order. The `orderitems` are associated with 
each order by `orderid`, and so the additional join uses that field to 
associate the joined customer-orders table with the `orderitems` table.

The second coding style takes advantage of Python's support for customizing 
the behavior of arithmetic operators. A natural operator for joining two 
tables would be the '+' operator.  To complete the join specification, we 
need not only the tables to be joined (the left and right terms of the '+' 
operation), but also the attributes to use to know which objects of each 
table to join together.  To support this, tables have the join_on() method, 
which return a `JoinTerm` object:

```python
customers.join_on("custid") + orders.join_on("custid")
```

This returns a join expression, which when called, performs the join and 
returns the data as a new `Table`:

```python
customerorders = (customers.join_on("custid") + orders.join_on("custid"))()
```

JoinTerms can be added to tables directly when the join table and the added
table are to join using the same attribute name.  The 3-table join above
can be written as:

```python
customerorderitems = ((customers.join_on("custid") 
                      + orders 
                      + orderitems.join_on("orderid"))())
```

A classic example of performing an outer join is, given a table of students
and a table of student->course registrations, find the students who
are not registered for any courses. The solution is to perform an outer
join, and select those students where their course registration is NULL.

Here is how that looks with littletable:

```python
# define student and registration data
students = lt.Table().csv_import("""\
student_id,name
0001,Alice
0002,Bob
0003,Charlie
0004,Dave
0005,Enid
""")

registrations = lt.Table().csv_import("""\
student_id,course
0001,PSYCH101
0001,CALC1
0003,BIO200
0005,CHEM101
""")

# perform outer join and show results:    
non_reg = students.outer_join(lt.Table.RIGHT_OUTER_JOIN, 
                              registrations, 
                              student_id="student_id").where(course=None)
non_reg.present()
print(list(non_reg.all.name))
```
    
Displays:

    +----------------------------+
    | Student_Id | Name | Course |
    |------------+------+--------|
    | 0002       | Bob  | None   |
    | 0004       | Dave | None   |
    +----------------------------+
    ['Bob', 'Dave']



Pivoting a table
----------------
Pivoting is a useful function for extracting overall distributions of values 
within a table. Tables can be pivoted on 1, 2, or 3 attributes. The pivot 
tallies up the number of objects in a given table with each of the different
key values.  A single attribute pivot gives the same results as a histogram - 
each key for the attribute is given, along with the count of objects having 
that key value.  (Of course, pivots are most interesting for attributes 
with non-unique indexes.)

Pivoting on 2 attributes extends the concept, getting the range of key values 
for each attribute, and then tallying the number of objects containing each 
possible pair of key values. The results can be reported as a two-dimensional 
table, with the primary attribute keys down the leftmost column, and the 
secondary attribute keys as headers across the columns.  Subtotals will also 
be reported at the far right column and at the bottom of each column.


littletable and pandas
----------------------
The `pandas` package is a mature and powerful data analysis module for Python,
with extensive analytical, statistical, and query features. It supports a 
vectorized approach to many common data operations, so that operations on an
entire column of values can be done very quickly.

However, the `pandas` package is rather heavyweight, and for simple table
operations (such as reading and accessing values in a CSV file), it may be
overpowered or overcomplicated to install and learn.

`littletable` is lightweight in contrast, and, as a pure Python module, is 
not optimized for vector operations on its "columns". As a single module
Python file, its installation and distribution can be very simple, so 
well suited to small data projects. Its design philosophy is to make simple 
tasks (such as CSV import and tabular display) simple, make some difficult 
tasks (such as pivot and join) possible, but leave the really difficult 
tasks to other packages, such as `pandas`.

`littletable` is useful even with tables of up to 1 or 2 million rows (especially 
with defined indexes); `pandas` can handle much larger datasets, and so is 
more suited to large Data Science projects.

Use `pandas` if:
- your data set is large
- your problem involves extensive vector operations on entire columns
  of data in your table
- your problem is simple (CSV import), but you already have installed
  `pandas` and are familiar with its usage

Consider `littletable` if:
- you do not already have `pandas` installed, or are unfamiliar with using
  DataFrames
- your dataset is no more than 1-2 million rows
- your data needs are simple CSV import/export, filtering, sorting,
  join/pivot, presentation


littletable and SQLite
----------------------
`littletable` and `SQLite` have many similarities in intent; lightweight,
easy to deploy and distribute. But `SQLite` requires some of the same 
schema design steps and SQL language learning of any relational database.
Mapping from Python program objects to database tables often requires use
of an ORM like `SQLAlchemy`. `littletable` allows you to store your Python
objects directly in a Table, and query against it using similar ORM-style
methods.

`littletable` might even be useful as a prototyping tool, to work through
some data modeling and schema design concepts before making a commitment
to a specific schema of tables and indexes.


Some simple littletable recipes
-------------------------------

- Find objects with NULL attribute values (an object's attribute is considered 
  NULL if the object does not have that attribute, or if its value is None or ""):

  ```python
        table.where(keyattr=Table.is_null())
  ```
    

- Histogram of values of a particular attribute:

  ```python
  # returns a table
  table.pivot(attribute).summary_counts()
  ```
   or
  ```python
  # prints the values to stdout in tabular form
  table.pivot(attribute).dump_counts()
  ```


- Get a list of all key values for an indexed attribute:

  ```python
  customers.by.zipcode.keys()
  ```


- Get a list of all values for any attribute:

  ```python
  list(customers.all.first_name)
  
  # or get just the unique values
  list(customers.all.first_name.unique)
  ```


- Get a count of entries for each key value:

  ```python
  customers.pivot("zipcode").dump_counts()
  ```

- Sort table by attribute x

  ```python
  employees.sort("salary")
  
  # sort in descending order
  employees.sort("salary desc")
  ```

- Sorted table by primary attribute x, secondary attribute y

  ```python
  sales_employees = employees.where(dept="Sales").sort("salary,commission")
  ```

  or

  ```python
  employees.create_index("dept")
  sales_employees = employees.by.dept["Sales"].sort("salary,commission")
  ```

- Get top 5 objects in table by value of attribute x
  ```python
  # top 5 sales employees
  employees.where(dept="Sales").sort("sales desc")[:5]
  ```

- Find all employees whose first name starts with "X"

  ```python
  employees.where(first_name=Table.startswith("X"))
  ```
