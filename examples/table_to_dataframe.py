#
# table_to_dataframe.py
#
# Short example showing how to create a pandas.DataFrame from
# a littletable.Table.
#
import itertools

import littletable as lt


# make a Table
counter = itertools.count()
labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
tbl = lt.Table().insert_many(
    (dict(zip("abc", (next(counter), next(counter), next(counter),)))
     | {"label": label}) for label in labels
)

# pretty output of tbl.info()
lt.Table().insert_many(
    {'property': k, 'value': v} for k, v in tbl.info().items()
)("tbl.info()").present()

# print first 5 rows starting table
tbl[:5].present()

# create DataFrame using extracted values with fieldnames as columns
df = tbl.as_dataframe()

# print first 5 rows of df
print(df[:5])
