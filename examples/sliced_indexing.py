# sliced_indexing.py
#
# Demonstration of using slices on indexed attributes to do range filtering.
# Even non-integer types are supported (as long as they support >= and <
# operations).
#
import littletable as lt
import textwrap
import datetime


def main():

    # read data from US place names CSV file
    transforms = {
        'pop': int,
        'elev': lambda s: int(float(s)*3.28084),
        'lat': float,
        'long': float,
    }
    us_ppl = lt.Table().csv_import(
        "examples/us_ppl.csv",
        transforms=transforms
    ).select("id name state elev lat long pop")

    print(us_ppl.info())
    us_ppl.create_index("elev")

    test = "elev < 0"
    us_ppl.by.elev[:0](f"{test} (sliced)")[:100].present()

    test = "elev >= 1000"
    us_ppl.by.elev[1000:](f"{test} (sliced)").sort("elev desc")[:100].present()

    test = "0 <= elev < 100"
    us_ppl.by.elev[0:100](f"{test} (sliced)")[:100].present()

    # slice using non-integer types

    us_ppl.create_index("name")
    a_ppl_slice = us_ppl.by.name["A":"C"]
    a_ppl_slice[:100].present()

    a_ppl_slice = us_ppl.by.name["Z":]
    a_ppl_slice[:100].present()

    # load some sales data from Acme, Inc.
    sales_data = textwrap.dedent("""\
        date,customer,sku,qty
        2000/01/01,0020,ANVIL-001,1
        2000/01/01,0020,BRDSD-001,5
        2000/02/15,0020,BRDSD-001,5
        2000/03/31,0020,BRDSD-001,5
        2000/03/31,0020,MAGNT-001,1
        2000/04/01,0020,ROBOT-001,1
        2000/04/15,0020,BRDSD-001,5
        1900/02/29,0020,BRDSD-001,5
        """)

    # load data from CSV, converting dates to datetime.date
    transforms = {'date': lt.Table.parse_date("%Y/%m/%d"),
                  'qty': int}
    sales = lt.Table("All 2000 Sales").csv_import(
        sales_data,
        transforms=transforms,
    )
    sales.present()

    # get sales from the first quarter only
    sales.create_index("date")
    jan_01 = datetime.date(2000, 1, 1)
    apr_01 = datetime.date(2000, 4, 1)
    first_qtr_sales = sales.by.date[jan_01: apr_01]("2000 Q1 Sales")
    first_qtr_sales.present()

    # load data from CSV, leave dates as strings (still sortable when in YYYY/MM/DD form)
    transforms = {'qty': int}
    sales = lt.Table("All 2000 Sales").csv_import(
        sales_data,
        transforms=transforms,
    )

    # get sales from the first quarter only
    sales.create_index("date")
    first_qtr_sales = sales.by.date["2000/01/01": "2000/04/01"]("2000 Q1 Sales")
    first_qtr_sales.present()


if __name__ == '__main__':
    main()
