#
# zen_of_python_pep20.py
#
# read the Zen of Python into a littletable.Table
#
from contextlib import redirect_stdout
import io
import littletable as lt

# get the Zen of Python content
import_this_stdout = io.StringIO()
with redirect_stdout(import_this_stdout):
    import this
import_this_lines = import_this_stdout.getvalue().splitlines()[2:]

# load into a Table and present
pep20 = lt.Table().insert_many(
    {'id': num, 'zen': zen}
    for num, zen in enumerate(import_this_lines, start=1)
)("The Zen of Python")
pep20.present()

# use text search for "better"
pep20.create_search_index("zen")
for entry in pep20.search.zen("better"):
    print(entry.zen)
