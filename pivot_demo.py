#
# pivot_demo.py
# 
# Using littletable, performs summaries on one and two attribute pivots
# of a table of US populated place names.
#
# Uses data excerpted from the US.TXT geographic names data file, provided
# under Creative Commons Attribution 3.0 License by GeoNames (www.geonames.org)
#
# Copyright (c) 2011  Paul T. McGuire
#
from littletable import Table

# import from csv, convert elevation from meters to feet
places = Table().csv_import('us_ppl.csv', 
                            transforms={'elev':lambda s:int(s)*33/10, 
                                        'pop':int})

# add computed field, elevation rounded down by 1000's
places.add_field('elev2', lambda x: int(x.elev/1000)*1000, 0)

# create indexes
places.create_index('state')
places.create_index('elev2')

print("summarize population by state")
piv = places.pivot('state')
pplByState = piv.as_table(sum, 'pop').sort('pop desc')
for rec in pplByState:
    print(rec.state, rec.pop)
piv.dump_counts(count_fn=lambda recs:sum(r.pop for r in recs))

print('')
print("summarize population by elevation")
piv = places.pivot('elev2')
pplByElev = piv.as_table(sum, 'pop')
for rec in pplByElev:
    print(rec.elev2, rec.pop)
piv.dump_counts(count_fn=lambda recs:sum(r.pop for r in recs))

print('')
print("summarize population by state and elevation")
piv = places.pivot('state elev2')
alaskan_locns_3k_to_4k_ft = piv['AK'][3000]
for rec in alaskan_locns_3k_to_4k_ft:
    print(rec)
print('')

piv.dump_counts(count_fn=lambda recs:sum(r.pop for r in recs))
print('')

pplByElev = piv.as_table(sum, 'pop')
for rec in pplByElev[:100]:
    print(rec.state, rec.elev2, rec.pop)

# find average elevation of person by state
print("Average elevation of each person by state")
piv = places.pivot('state')
piv.dump_counts(count_fn=lambda recs: int(sum(r.pop*r.elev for r in recs)/sum(r.pop for r in recs)))
