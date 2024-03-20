# explore_unicode.py
#
# Read and process the standard Unicode table (from http://www.unicode.org/L2/L1999/UnicodeData.html)
#
#   Field    Title                             Normative/Informative     Desc
#  ──────────────────────────────────────────────────────────────────────────────────────────────────────
#       0    Code value                        normative                 Code value in 4-digit
#                                                                        hexadecimal format.
#       1    Character name                    normative                 These names match exactly the
#                                                                        names published in Chapter 7 of
#                                                                        the Unicode Standard, Version
#                                                                        2.0, except for the two
#                                                                        additional characters.
#       2    General category                  normative / informative   This is a useful breakdown into
#                                                                        various "character types" which
#                                                                        can be used as a default
#                                                                        categorization in
#                                                                        implementations. See below for
#                                                                        a brief explanation.
#       3    Canonical combining classes       normative                 The classes used for the
#                                                                        Canonical Ordering Algorithm in
#                                                                        the Unicode Standard. These
#                                                                        classes are also printed in
#                                                                        Chapter 4 of the Unicode
#                                                                        Standard.
#       4    Bidirectional category            normative                 See the list below for an
#                                                                        explanation of the
#                                                                        abbreviations used in this
#                                                                        field. These are the categories
#                                                                        required by the Bidirectional
#                                                                        Behavior Algorithm in the
#                                                                        Unicode Standard. These
#                                                                        categories are summarized in
#                                                                        Chapter 3 of the Unicode
#                                                                        Standard.
#       5    Character decomposition mapping   normative                 In the Unicode Standard, not
#                                                                        all of the mappings are full
#                                                                        (maximal) decompositions.
#                                                                        Recursive application of
#                                                                        look-up for decompositions
#                                                                        will, in all cases, lead to a
#                                                                        maximal decomposition. The
#                                                                        decomposition mappings match
#                                                                        exactly the decomposition
#                                                                        mappings published with the
#                                                                        character names in the Unicode
#                                                                        Standard.
#       6    Decimal digit value               normative                 This is a numeric field. If the
#                                                                        character has the decimal digit
#                                                                        property, as specified in
#                                                                        Chapter 4 of the Unicode
#                                                                        Standard, the value of that
#                                                                        digit is represented with an
#                                                                        integer value in this field
#       7    Digit value                       normative                 This is a numeric field. If the
#                                                                        character represents a digit,
#                                                                        not necessarily a decimal
#                                                                        digit, the value is here. This
#                                                                        covers digits which do not form
#                                                                        decimal radix forms, such as
#                                                                        the compatibility superscript
#                                                                        digits
#       8    Numeric value                     normative                 This is a numeric field. If the
#                                                                        character has the numeric
#                                                                        property, as specified in
#                                                                        Chapter 4 of the Unicode
#                                                                        Standard, the value of that
#                                                                        character is represented with
#                                                                        an integer or rational number
#                                                                        in this field. This includes
#                                                                        fractions as, e.g., "1/5" for
#                                                                        U+2155 VULGAR FRACTION ONE
#                                                                        FIFTH Also included are
#                                                                        numerical values for
#                                                                        compatibility characters such
#                                                                        as circled numbers.
#       9    Mirrored                          normative                 If the character has been
#                                                                        identified as a "mirrored"
#                                                                        character in bidirectional
#                                                                        text, this field has the value
#                                                                        "Y"; otherwise "N". The list of
#                                                                        mirrored characters is also
#                                                                        printed in Chapter 4 of the
#                                                                        Unicode Standard.
#       10   Unicode 1.0 Name                  informative               This is the old name as
#                                                                        published in Unicode 1.0. This
#                                                                        name is only provided when it
#                                                                        is significantly different from
#                                                                        the Unicode 3.0 name for the
#                                                                        character.
#       11   10646 comment field               informative               This is the ISO 10646 comment
#                                                                        field. It is in parantheses in
#                                                                        the 10646 names list.
#       12   Uppercase mapping                 informative               Upper case equivalent mapping.
#                                                                        If a character is part of an
#                                                                        alphabet with case
#                                                                        distinctions, and has an upper
#                                                                        case equivalent, then the upper
#                                                                        case equivalent is in this
#                                                                        field. See the explanation
#                                                                        below on case distinctions.
#                                                                        These mappings are always
#                                                                        one-to-one, not one-to-many or
#                                                                        many-to-one. This field is
#                                                                        informative.
#       13   Lowercase mapping                 informative               Similar to Uppercase mapping
#       14   Titlecase mapping                 informative               Similar to Uppercase mapping
# """
#
import re
import littletable as lt


fieldnames = [
    "code_value_hex",
    "name",
    "category",
    "combining_classes",
    "bidirectional_category",
    "char_decomposition_mapping",
    "decimal_digit_value",
    "digit_value",
    "numeric_value",
    "mirrored",
    "unicode_1_name",
    "iso10646_comment",
    "uppercase_hex",
    "lowercase_hex",
    "titlecase_hex",
]
unicode_url = "https://www.unicode.org/Public/3.2-Update/UnicodeData-3.2.0.txt"
unicode_file = "unicode_320.txt.zip"
unicode = lt.Table().csv_import(
    unicode_file,
    delimiter=";",
    transforms={
        "decimal_digit_value": int,
        "digit_value": int,
        "numeric_value": int,
    },
    fieldnames=fieldnames,
)
unicode.add_field("code_value", lambda r: int(r.code_value_hex, 16))
unicode.add_field("uppercase", lambda r: int(r.uppercase_hex, 16))
unicode.add_field("lowercase", lambda r: int(r.lowercase_hex, 16))
unicode.add_field("titlecase", lambda r: int(r.titlecase_hex, 16))
unicode.add_field("character", lambda r: chr(r.code_value))
unicode.add_field("upper_char", lambda r: chr(r.uppercase))
unicode.add_field("lower_char", lambda r: chr(r.lowercase))
unicode.add_field("title_char", lambda r: chr(r.titlecase))
unicode.add_field("is_identifier", lambda r: r.character.isidentifier())

unicode.create_index("code_value_hex", unique=True)
unicode.create_index("code_value", unique=True)

#
# Explore some interesting groups of symbols in the Unicode set
#


def present_symbol_group(
    start_str: str, title: str, source_table: lt.Table = unicode
) -> lt.Table:
    """
    Function to search for Unicode characters that match a starting string, and
    presents a table showing name, character, and decimal code value
    """
    tbl = source_table.where(name=lt.Table.startswith(start_str))(title)
    tbl = tbl.select("name character code_value code_value_hex")
    tbl.present(
        caption="Total {} symbols".format(len(tbl)),
        caption_justify="left",
    )
    return tbl


def present_symbol_group_contains_word(
    word_str: str, title: str, source_table: lt.Table = unicode
) -> lt.Table:
    """
    Function to search for Unicode characters that match a starting string, and
    presents a table showing name, character, and decimal code value
    """
    # DEPRECATED FORM
    # tbl = source_table.where(name=lt.Table.re_match(rf".*\b{word_str}\b"))(title)

    # NEW FORM
    contains_word = re.compile(rf"\b{word_str}\b").search
    tbl = source_table.where(name=contains_word)(title)

    tbl = tbl.select("name character code_value code_value_hex")
    tbl.present(
        caption="Total {} symbols".format(len(tbl)),
        caption_justify="left",
    )
    return tbl


# display the characters of the I Ching
i_ching = present_symbol_group("HEXAGRAM FOR", "I Ching")

# display the characters of the Tai Xuan Jing
tai_xuan_jing = present_symbol_group("TETRAGRAM FOR", "Tai Xuan Jing")

# display all the Roman numerals
numerics = unicode.where(numeric_value=lt.Table.ne(None)).orderby("numeric_value")
roman_numerals = present_symbol_group("ROMAN NUMERAL", "Roman Numerals", numerics)

# display all Braille characters
braille = present_symbol_group("BRAILLE PATTERN", "Braille")

# display all Box Drawing characters
box_drawing = present_symbol_group(
    "BOX DRAWINGS", "Box Drawing", unicode.where(code_value=lt.Table.lt(10000))
)

# clock faces
clock_faces = present_symbol_group("CLOCK FACE", "Clock Faces")

# die faces
die_faces = present_symbol_group("DIE FACE", "Die Faces")

# chess pieces
chess_pieces = present_symbol_group_contains_word(r"^(WHITE|BLACK) CHESS \w+$", "Chess Pieces")

faces = present_symbol_group_contains_word(r"FACE", "Faces")

dots = present_symbol_group_contains_word(r"DOT", "Dots")

arrows = present_symbol_group_contains_word(r"ARROW", "Arrows")
