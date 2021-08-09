#
# dune_casts.py
#
# Use join() to create a consolidated table of the casts of Dune,
# from the 1984 movie, the 2000 mini-series, and the 2021 movie.
#
import littletable as lt
from operator import attrgetter

dune_casts_csv = """\
character,actor,year
Lady Jessica Atreides,Francesca Annis,1984
Piter De Vries,Brad Dourif,1984
Padishah Emperor Shaddam IV,José Ferrer,1984
Shadout Mapes,Linda Hunt,1984
Thufir Hawat,Freddie Jones,1984
Duncan Idaho,Richard Jordan,1984
Paul Atreides,Kyle MacLachlan,1984
Princess Irulan,Virginia Madsen,1984
Reverend Mother Ramallo,Silvana Mangano,1984
Stilgar,Everett McGill,1984
Baron Vladimir Harkonnen,Kenneth McMillan,1984
Reverend Mother Gaius Helen Mohiam,Siân Phillips,1984
Duke Leto Atreides,Jürgen Prochnow,1984
Glossu Beast Rabban,Paul L. Smith,1984
Gurney Halleck,Patrick Stewart,1984
Feyd-Rautha Harkonnen,Sting,1984
Dr. Yueh,Dean Stockwell,1984
Dr. Liet-Kynes,Max von Sydow,1984
Alia Atreides,Alicia Witt,1984
Chani,Sean Young,1984
Otheym,Honorato Magaloni,1984
Jamis,Judd Omen,1984
Harah,Molly Wryn,1984
character,actor
Duke Leto Atreides,William Hurt,2000
Paul Atreides,Alec Newman,2000
Lady Jessica Atreides,Saskia Reeves,2000
Gurney Halleck,P.H. Moriarty,2000
Baron Vladimir Harkonnen,Ian McNeice,2000
Feyd-Rautha Harkonnen,Matt Keeslar,2000
Glossu Beast Rabban,László I. Kish,2000
Padishah Emperor Shaddam IV,Giancarlo Giannini,2000
Princess Irulan,Julie Cox,2000
Stilgar,Uwe Ochsenknecht,2000
Reverend Mother Gaius Helen Mohiam,Zuzana Geislerová,2000
Alia Atreides,Laura Burton,2000
Duncan Idaho,James Watson,2000
Chani,Barbora Kodetová,2000
Otheym,Jakob Schwarz,2000
Dr. Liet-Kynes,Karel Dobrý,2000
Thufir Hawat,Jan Vlasák,2000
Dr. Yueh,Robert Russell,2000
Piter De Vries,Jan Unger,2000
Jamis,Christopher Lee Brown,2000
Shadout Mapes,Jaroslava Siktancova,2000
Reverend Mother Ramallo,Drahomíra Fialková,2000
Young Mother Ramallo,Petra Spindlerová,2000
Harah,...,2000
Duncan Idaho,Jason Momoa,2021
Paul Atreides,Timothée Chalamet,2021
Chani,Zendaya,2021
Lady Jessica Atreides,Rebecca Ferguson,2021
Gurney Halleck,Josh Brolin,2021
Glossu Beast Rabban,Dave Bautista,2021
Duke Leto Atreides,Oscar Isaac,2021
Stilgar,Javier Bardem,2021
Piter De Vries,David Dastmalchian,2021
Baron Vladimir Harkonnen,Stellan Skarsgård,2021
Reverend Mother Gaius Helen Mohiam,Charlotte Rampling,2021
Thufir Hawat,Stephen McKinley Henderson,2021
Dr. Yueh,Chen Chang,2021
Dr. Liet-Kynes,Sharon Duncan-Brewster,2021
Jamis,Babs Olusanmokun,2021
Harah,Gloria Obianyo,2021
Padishah Emperor Shaddam IV,...,2021
Shadout Mapes,...,2021
Princess Irulan,...,2021
Reverend Mother Ramallo,...,2021
Feyd-Rautha Harkonnen,...,2021
Alia Atreides,...,2021
Otheym,...,2021
"""

dune_casts = lt.Table("dune_1984").csv_import(dune_casts_csv).create_index("character")
dune_1984 = dune_casts.where(year="1984").add_field("actor (1984)", attrgetter("actor"))
dune_2000 = dune_casts.where(year="2000").add_field("actor (2000)", attrgetter("actor"))
dune_2021 = dune_casts.where(year="2021").add_field("actor (2021)", attrgetter("actor"))

join = dune_1984.join_on("character") + dune_2000 + dune_2021
dune_combined = join()("Dune Casts (combined)")
dune_combined.present(
    fields=["character", "actor (1984)", "actor (2000)", "actor (2021)"]
)
