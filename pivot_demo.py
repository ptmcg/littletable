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
from urllib import urlopen

places = Table()

# use one of these to read data either from my public Dropbox, or
# download the data file locally and just read from the local file
#~ data_source = urlopen('http://dl.dropbox.com/u/19965781/us_ppl.csv')
data_source = 'US_PPL.CSV'

# import from csv, convert elevation from meters to feet
places.csv_import(data_source, 
                    transforms={'elev':lambda s:int(s)*33/10, 'pop':int})
places.create_index('state')

# add computed field, elevation rounded down by 1000's
places.addfield('elev2', lambda x: x.elev/1000*1000, 0)
places.create_index('elev2')

print "summarize population by state"
pplByState = places.pivot('state').summary_counts(sum, 'pop')
for rec in pplByState:
    print rec.state, rec.pop

print
print "summarize population by elevation"
pplByElev = places.pivot('elev2').summary_counts(sum, 'pop')
for rec in pplByElev:
    print rec.elev2, rec.pop

print
print "summarize population by state and elevation"
pplByElev = places.pivot('state elev2').summary_counts(sum, 'pop')
for rec in pplByElev[:100]:
    print rec.state, rec.elev2, rec.pop

