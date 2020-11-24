How to Use littletable
======================

  * [Introduction](#introduction)
  * [Creating a table](#creating-a-table)
  * [Inserting objects](#inserting-objects)
  * [Importing data from CSV files](#importing-data-from-csv-files)
  * [Tabular output](#tabular-output)
  * [DataObjects](#dataobjects)
  * [Removing objects](#removing-objects)
  * [Indexing attributes](#indexing-attributes)
  * [Querying with indexed attributes](#querying-with-indexed-attributes)
  * [Querying for exact matching attribute values](#querying-for-exact-matching-attribute-values)
  * [Querying for attribute value ranges](#querying-for-attribute-value-ranges)
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

    t = Table()
    
If you want, you can name the table at creation time, or any time later. 

    t = Table("customers")
    
or
    
    t = Table()
    t("customers")

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

    t.insert(obj)
    t.insert_many(objlist)

Performance tip: Calling `insert_many()` with a list of objects will perform better than calling
`insert()` in a loop.

`littletable` supports records that are user-defined types (including those defined
using `__slots__`), `namedtuple`s, `SimpleNamespace`s. Python `dict`s can be used if
they are converted to `SimpleNamespace`s or `littletable.DataObject`s.


Importing data from CSV files
-----------------------------
You can easily import a CSV file into a `Table` using `Table.csv_import()`:

    t = Table().csv_import("my_data.csv")

In place of a local file name, you can also specify an HTTP url:

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    iris_table = Table('iris').csv_import(url)

You can also directly import CSV data as a string:

    catalog_data = """\
    sku,description,unitofmeas,unitprice
    BRDSD-001,Bird seed,LB,3
    BBS-001,Steel BB's,LB,5
    MGNT-001,Magnet,EA,8"""

    catalog = Table("catalog")
    catalog.create_index("sku", unique=True)
    catalog.csv_import(catalog_data, transforms={'unitprice': int})

If you are working with a very large CSV file and just trying to see 
what the structure is, add `limit=100` to only read the first 100 rows.

You can also pre-screen data as it is read from the input file by passing
a `filters={attr: filter_fn, ...}` argument. Each filter function is called
on the newly-read object _before_ it is added to the table. `filter_fn` 
can be any function that takes a single argument of the type of the given
attribute, and returns `True` or `False`. If `False`, the record does not get
added to the table.

    # import only the first 100 items that match the filter
    # (product_category == "Home and Garden")
    catalog.csv_import(catalog_data, 
                       filters={"product_category": Table.eq("Home and Garden")},
                       limit=100)

Since CSV files do not keep any type information, `littletable` will use its 
own `DataObject` type for imported records. You can specify your own type by 
passing `row_class=MyType` to `csv_import`. The type must be initializable
using the form `MyType(**attributes_dict)`. `namedtuples` and `SimpleNamespace`
both support this form.

Performance tip: For very large files, it is faster to load data using
a `namedtuple` than to use the default `DataObject` class. Get the fields using
a 10-item import using `limit=10`, and then getting the fields from 
`table.info()["fields"]`.

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


Tabular output
--------------
To produce a nice tabular output for a table, you can use the embedded support for
the `rich` module, `as_html()` in Jupyter Notebook, or the `tabulate` module:

- Using `table.present()` (implemented using `rich`; `present()` accepts `rich` Table 
  keyword args):

      table(title_str).present(fields=["col1", "col2", "col3"])

  or

      # use select() to limit the columns to be shown
      table.select("col1 col2 col3")(title_str).present(caption="caption text")

- Using `Jupyter Notebook`:

      from IPython.display import HTML, display
      display(HTML(table.as_html()))

- Using `tabulate`:

      # use map(vars, table) to get each table record as a dict, then pass to tabulate with
      # headers="keys" to auto-define headers
      print(tabulate(map(vars, table), headers="keys"))


DataObjects
-----------
If your program does not have a type for inserting into your table, or if your
records are Python `dict`s, you can use the `littletable` type `DataObject`:

    t.insert(DataObject(name="Alice", age=20))
    bob = {"name": "Bob", "age": 19}
    t.insert(DataObject(**bob))

or if using a Python >= 3.3, you can use `types.SimpleNamespace`:

    from types import SimpleNamespace
    bob = {"name": "Bob", "age": 19}
    t.insert(SimpleNamespace(**bob))


Removing objects
----------------
Objects can be removed individually or by passing a list (or `Table`) of
objects:

    t.remove(obj)
    t.remove_many(objlist)
    t.remove_many(t.where(a=100))
    

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

    employees.create_index('employee_id', unique=True)
    employees.create_index('zipcode')

    # unique indexes return a single object
    print(employees.by.employee_id["D1729"].name)
    
    # non unique indexes return a new Table
    for emp in employees.by.zipcode["12345"]:
        print(e.name)


Querying with indexed attributes
--------------------------------

If accessing a table using a unique index, giving a key value will 
return the single matching record, or raise `KeyError`.

    employees.by.employee_id['00086']
    employees.by.employee_id['invalid_id']
    #    raises KeyError: "no such value 'invalid_id' in index 'employee_id'"


If accessing a table using a non-unique index, will return a new `Table` 
containing all matching records. If there are no matching records, the 
returned table will be empty.

    employees.by.state['CA']
    employees.by.dept['Sales']
    employees.by.dept['Salex']  # no such department
    #    returns empty table


Querying for exact matching attribute values
--------------------------------------------
Calling `Table.where()` with named attributes will return a Table of
all records matching all the arguments:

    employees.where(zipcode="12345", title="Manager")    
    student.where(**{"class":"Algebra"})
    

Querying for attribute value ranges
-----------------------------------
`Table.where()` will also accept a callable that takes a record and
returns a bool to indicate if the record is a match:

    employees.where(lambda emp: emp.salary > 50000)

`littletable` also includes comparators to make range-checking easier to
write:

    # Comparators are:
    # - Table.lt        attr=Table.lt(100)             attr < 100
    # - Table.le        attr=Table.le(100)             attr <= 100
    # - Table.gt        attr=Table.gt(100)             attr > 100
    # - Table.ge        attr=Table.ge(100)             attr >= 100
    # - Table.eq        attr=Table.eq(100)             attr == 100
    # - Table.ne        attr=Table.ne(100)             attr != 100
    # - Table.between   attr=Table.between(100, 200)   100 < attr < 200
    # - Table.within    attr=Table.within(100, 200)    100 <= attr <= 200
    # - Table.in_range  attr=Table.in_range(100, 200)  100 <= attr < 200
    # - Table.is_in     attr=Table.is_in((1, 2, 3))    attr in (1,2,3)
    # - Table.not_in    attr=Table.not_in((1, 2, 3))   attr not in (1,2,3)

    employees.where(salary=Table.gt(50000))
    employees.where(dept=Table.is_in(["Sales", "Marketing"]))

    jan_01 = date(2000, 1, 1)
    mar_31 = date(2000, 3, 31)
    apr_01 = date(2000, 4, 1)
    
    first_qtr_sales = sales.where(date=Table.within(jan_01, mar_31))
    first_qtr_sales = sales.where(date=Table.in_range(jan_01, apr_01))

Comparators can also be used as filter functions for import methods.


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

    t1 = lt.Table()
    t1.csv_import("""\
    a,b,c
    100,101,102
    110,220,99
    108,130,109""", transforms=dict(a=int, b=int, c=int))
    
    t1_stats = t1.stats()
    t1_stats.present(box=box.ASCII)
    print(t1_stats.by.name["a"].mean)
    
    #    +-----------------------------------------------------------------------------+
    #    | Name | Count | Min | Max |           Mean |       Variance |        Std_Dev |
    #    |------+-------+-----+-----+----------------+----------------+----------------|
    #    | a    |     3 | 100 | 110 |          106.0 |             28 | 5.29150262212  |
    #    | b    |     3 | 101 | 220 | 150.333333333  | 3850.33333333  | 62.0510542483  |
    #    | c    |     3 |  99 | 109 | 103.333333333  | 26.3333333333  | 5.13160143944  |
    #    +-----------------------------------------------------------------------------+
    #    106.0

    
    t1_stats = t1.stats(by_field=False)
    t1_stats.present(box=box.ASCII)
    print(t1_stats.by.stat["mean"].a)

    #    +------------------------------------------------------------------------+
    #    | Stat     |                 A |                  B |                  C |
    #    |----------+-------------------+--------------------+--------------------|
    #    | count    |                 3 |                  3 |                  3 |
    #    | min      |               100 |                101 |                 99 |
    #    | max      |               110 |                220 |                109 |
    #    | mean     |             106.0 | 150.33333333333334 | 103.33333333333333 |
    #    | variance |                28 | 3850.3333333333335 | 26.333333333333332 |
    #    | std_dev  | 5.291502622129181 | 62.051054248363364 |  5.131601439446884 |
    #    +------------------------------------------------------------------------+
    #    106.0


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

    columns = [
        ("id_no", 0, ),
        ("name", 4, ),
        ("address", 21, ),
        ("city", 42, ),
        ("state", 56, 58, ),
        ("tech_skill_score", 59, None, float),
        ]

And use a `FixedWidthReader` to read the file and pass a list of 
dicts to a `Table.insert_many`:

    characters = lt.Table()
    reader = lt.FixedWidthReader(columns, "cartoon_characters.txt")
    characters.insert_many(lt.DataObject(**rec)
                           for rec in reader)

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

    customers.join(orders, custid="custid")

creates an inner join between the table of customers and the table of 
their respective orders, joining on both tables' `custid` attributes.

More than 2 tables can be joined in succession, since the result of a 
join is itself a `Table`:

    customers.join(orders, custid="custid").join(orderitems, orderid="orderid")
    
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

    customers.join_on("custid") + orders.join_on("custid")

This returns a join expression, which when called, performs the join and 
returns the data as a new `Table`:

    customerorders = (customers.join_on("custid") + orders.join_on("custid"))()

JoinTerms can be added to tables directly when the join table and the added
table are to join using the same attribute name.  The 3-table join above
can be written as:

    customerorderitems = ((customers.join_on("custid") 
                          + orders 
                          + orderitems.join_on("orderid"))())

A classic example of performing an outer join is, given a table of students
and a table of student->course registrations, find the students who
are not registered for any courses. The solution is to perform an outer
join, and select those students where their course registration is NULL.

Here is how that looks with littletable:

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
  NULL if the object does not have that attribute, or if its value is None):

      table.where(lambda rec: getattr(rec, keyattr, None) is None)
    

- Histogram of values of a particular attribute:

      (returns a table)
      table.pivot(attribute).summary_counts()

  or
  
      (prints the values to stdout in tabular form)
      table.pivot(attribute).dump_counts()


- Get a list of all key values for an indexed attribute:

      customers.zipcode.keys()


- Get a count of entries for each key value:

      customers.pivot("zipcode").dump_counts()
    

- Sorted table by attribute x

      employees.sort("salary")
    

- Sorted table by primary attribute x, secondary attribute y

      sales_employees = employees.where(dept="Sales").sort("salary,commission")
      
  or

      employees.create_index("dept")
      sales_employees = employees.by.dept["Sales"].sort("salary,commission")

- Get top 5 objects in table by value of attribute x

      # top 5 sales employees
      employees.where(dept="Sales").sort("sales desc")[:5]
