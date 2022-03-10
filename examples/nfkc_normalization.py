# NFKC_normalization.py
#
# List out Unicode characters that normalize to the ASCII character set.
#
# Copyright Paul McGuire, 2022
#
import sys
import unicodedata

import littletable as lt

ALL_UNICODE_CHARS = False

# initialize accumulator for ASCII character to collect Unicode characters
# that normalize back to ASCII characters that can be used in Python
# identifiers (A-Z, a-z, 0-9, _, and ·)
_· = "·"
ident_chars = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
               "abcdefghijklmnopqrstuvwxyz"
               "0123456789_" + _·)
accum = {ch: [] for ch in ident_chars}

# build up accumulator by walking Unicode range
unicode_upper_limit = sys.maxunicode+1 if ALL_UNICODE_CHARS else 65536
for i in range(32, unicode_upper_limit):
    ch = chr(i)
    norm = unicodedata.normalize("NFKC", ch)
    if norm in accum:
        accum[norm].append(ch)

# convert accumulator to a littletable Table for presentation
normalizations = lt.Table()
for asc_char, normalizing_chars in accum.items():
    normalizations.insert_many(
        {"ASCII": asc_char,
         "ord": ord(asc_char),
         "Unicode": norm_char,
         "code_point": ord(norm_char),
         "name": unicodedata.name(norm_char),
         }
        for norm_char in normalizing_chars
    )
normalizations.sort("ASCII")
normalizations.present(groupby="ASCII ord")
