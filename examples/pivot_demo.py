#
# pivot_demo.py
#
# Using littletable, performs summaries on one and two attribute pivots
# of a table of US populated place names.
#
# Uses data excerpted from the US.TXT geographic names data file, provided
# under Creative Commons Attribution 3.0 License by GeoNames (www.geonames.org)
#
# Copyright (c) 2011, 2024  Paul T. McGuire
#
import littletable as lt
from pathlib import Path


this_dir = Path(__file__).parent

# import from csv, convert elevation from meters to feet
places: lt.Table = lt.csv_import(
    this_dir / "us_ppl.zip",
    transforms={
        "elev": lambda s: int(s) * 33 / 10,
        "pop": int
    },
)
print(places.info())

# add computed field, elevation rounded down by 1000's
places.compute_field("elev_000", lambda x: int(x.elev / 1000) * 1000, 0)

# create indexes for use by pivots
places.create_index("state")
places.create_index("elev_000")

print("summarize population by state")
piv = places.pivot("state")
ppl_by_state = piv.as_table(sum, "pop").orderby("pop desc")
for rec in ppl_by_state:
    print(rec.state, rec.pop)

print()
piv.dump_counts(count_fn=lambda recs: sum(r.pop for r in recs))

print()
print("summarize population by elevation")
piv = places.pivot("elev_000")
ppl_by_elev = piv.as_table(sum, "pop")
for rec in ppl_by_elev:
    print(rec.elev_000, rec.pop)

print()
print("summarize population by state and elevation")
piv = places.pivot("state elev_000")

# dump all the sum of all population attributes for each row in each subtable
piv.dump_counts(count_fn=lambda recs: sum(r.pop for r in recs))
print()

# select a subtable from the pivot table
sd0 = piv["SD"][0]("SD places below 1000 ft elev")
sd0.select("name pop elev elev_000").present()
print()

# pplByElev = piv.as_table(sum, "pop")
# for rec in pplByElev[:100]:
#     print(rec.state, rec.elev_000, rec.pop)

# find average elevation of person by state
print("Average elevation of each person by state")
piv = places.pivot("state")
piv.dump_counts(
    count_fn=lambda recs: int(
        sum(r.pop * r.elev for r in recs) / sum(r.pop for r in recs)
    )
)

low_liers = places.where(elev=lt.Table.le(20))
print(f"\nPopulation at or below 20 feet sea level: {sum(low_liers.all.pop):,}")