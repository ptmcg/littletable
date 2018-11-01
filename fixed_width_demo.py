import littletable as lt
import io

data = """\
0010GEORGE JETSON    12345 SPACESHIP ST   HOUSTON       TX 4.9
0020WILE E COYOTE    312 ACME BLVD        TUCSON        AZ 7.3
0030FRED FLINTSTONE  246 GRANITE LANE     BEDROCK       CA 2.6
0040JONNY QUEST      31416 SCIENCE AVE    PALO ALTO     CA 8.1
"""

columns = [
    ("id_no", 0, ),
    ("name", 4, ),
    ("address", 21, ),
    ("city", 42, ),
    ("state", 56, 58, ),
    ("tech_skill_score", 59, None, float),
    ]

characters_table = lt.Table().insert_many(lt.DataObject(**rec) for rec in
                                          lt.FixedWidthReader(columns, io.StringIO(data)))

print(len(characters_table))
print(characters_table[0])