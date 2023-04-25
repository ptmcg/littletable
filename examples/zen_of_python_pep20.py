import subprocess
import littletable as lt

proc = subprocess.run(['python', '-c', 'import this'], capture_output=True)
import_this_stdout = proc.stdout.decode().splitlines()[2:]

pep20 = lt.Table().insert_many(
    {'id': num, 'zen': zen}
    for num, zen in enumerate(import_this_stdout, start=1)
)("The Zen of Python")
pep20.present()

# use text search for "better"
pep20.create_search_index("zen")
for entry in pep20.search.zen("better"):
    print(entry[0].zen)
