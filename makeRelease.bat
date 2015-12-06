rmdir dist

if exist MANIFEST del MANIFEST
python setup.py sdist --formats=gztar,zip register

if exist MANIFEST del MANIFEST

python setup.py bdist_wheel
python setup.py bdist_wininst --target-version=2.6 --plat-name=win32 upload register
python setup.py bdist_wininst --target-version=2.7 --plat-name=win32 upload register
python setup.py bdist_wininst --target-version=3.0 --plat-name=win32 upload register
python setup.py bdist_wininst --target-version=3.1 --plat-name=win32 upload register
python setup.py bdist_wininst --target-version=3.2 --plat-name=win32 upload register
python setup.py bdist_wininst --target-version=3.3 --plat-name=win32 upload register
python setup.py bdist_wininst --target-version=3.4 --plat-name=win32 upload register