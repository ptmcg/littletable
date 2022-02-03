# NFKC_normalization.py
#
# List out Unicode characters that normalize to the ASCII character set.
#
# Copyright Paul McGuire, 2022
#
import unicodedata

import littletable as lt

# initialize accumulator for ASCII character to collect Unicode characters
# that normalize back to ASCII
accum = {ch: [] for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_0123456789"}

# build up accumulator by walking Unicode range
for i in range(32, 65536):  # sys.maxunicode+1):
    ch = chr(i)
    norm = unicodedata.normalize("NFKC", ch)
    if norm in accum:
        accum[norm].append(ch)

# convert accumulator to a littletable Table for presentation
normalization = lt.Table()
for a, norm in accum.items():
    normalization.insert_many(
        {"ASCII": a,
         "ord": ord(a),
         "Unicode": n,
         "code_point": ord(n),
         "name": unicodedata.name(n),
         }
        for n in norm
    )
normalization.present(groupby="ASCII ord")
