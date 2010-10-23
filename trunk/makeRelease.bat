rmdir dist

if exist MANIFEST del MANIFEST
python setup.py sdist --formats=gztar,zip
