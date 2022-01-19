#
# excel_data_types.py
#
# Demonstrate data type conversions done automatically when importing from Excel
#
import littletable as lt

xl = lt.Table().excel_import("../test/data_types.xlsx")

xl.present()
for row in xl:
    print(row.name, repr(row.value), type(row.value), row.type)
