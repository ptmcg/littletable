#
# dune_casts.py
#
# Use join() to create a consolidated table of the casts of Dune,
# from the 1984 movie, the 2000 mini-series, and the 2021 movie.
#
import littletable as lt

dune_1984_csv = """\
character,actor_1984
Lady Jessica Atreides,Francesca Annis
Piter De Vries,Brad Dourif
Padishah Emperor Shaddam IV,José Ferrer
Shadout Mapes,Linda Hunt
Thufir Hawat,Freddie Jones
Duncan Idaho,Richard Jordan
Paul Atreides,Kyle MacLachlan
Princess Irulan,Virginia Madsen
Reverend Mother Ramallo,Silvana Mangano
Stilgar,Everett McGill
Baron Vladimir Harkonnen,Kenneth McMillan
Reverend Mother Gaius Helen Mohiam,Siân Phillips
Duke Leto Atreides,Jürgen Prochnow
Glossu Beast Rabban,Paul L. Smith
Gurney Halleck,Patrick Stewart
Feyd-Rautha Harkonnen,Sting
Dr. Yueh,Dean Stockwell
Dr. Liet-Kynes,Max von Sydow
Alia Atreides,Alicia Witt
Chani,Sean Young
Otheym,Honorato Magaloni
Jamis,Judd Omen
Harah,Molly Wryn
"""

dune_2000_csv = """\
character,actor_2000
Duke Leto Atreides,William Hurt
Paul Atreides,Alec Newman
Lady Jessica Atreides,Saskia Reeves
Gurney Halleck,P.H. Moriarty
Baron Vladimir Harkonnen,Ian McNeice
Feyd-Rautha Harkonnen,Matt Keeslar
Glossu Beast Rabban,László I. Kish
Padishah Emperor Shaddam IV,Giancarlo Giannini
Princess Irulan,Julie Cox
Stilgar,Uwe Ochsenknecht
Reverend Mother Gaius Helen Mohiam,Zuzana Geislerová
Alia Atreides,Laura Burton
Duncan Idaho,James Watson
Chani,Barbora Kodetová
Otheym,Jakob Schwarz
Dr. Liet-Kynes,Karel Dobrý
Thufir Hawat,Jan Vlasák
Dr. Yueh,Robert Russell
Piter De Vries,Jan Unger
Jamis,Christopher Lee Brown
Shadout Mapes,Jaroslava Siktancova
Reverend Mother Ramallo,Drahomíra Fialková
Young Mother Ramallo,Petra Spindlerová
Harah,...
"""

dune_2021_csv = """\
character,actor_2021
Duncan Idaho,Jason Momoa
Paul Atreides,Timothée Chalamet
Chani,Zendaya
Lady Jessica Atreides,Rebecca Ferguson
Gurney Halleck,Josh Brolin
Glossu Beast Rabban,Dave Bautista
Duke Leto Atreides,Oscar Isaac
Stilgar,Javier Bardem
Piter De Vries,David Dastmalchian
Baron Vladimir Harkonnen,Stellan Skarsgård
Reverend Mother Gaius Helen Mohiam,Charlotte Rampling
Thufir Hawat,Stephen McKinley Henderson
Dr. Yueh,Chen Chang
Dr. Liet-Kynes,Sharon Duncan-Brewster
Jamis,Babs Olusanmokun
Harah,Gloria Obianyo
Padishah Emperor Shaddam IV,...
Shadout Mapes,...
Princess Irulan,...
Reverend Mother Ramallo,...
Feyd-Rautha Harkonnen,...
Alia Atreides,...
Otheym,...
"""

dune_1984 = (lt.Table("dune_1984").csv_import(dune_1984_csv)
                                  .create_index("character"))

dune_2000 = (lt.Table("dune_2000").csv_import(dune_2000_csv)
                                  .create_index("character"))

dune_2021 = (lt.Table("dune_2021").csv_import(dune_2021_csv)
                                  .create_index("character"))

join = dune_1984.join_on("character") + dune_2000 + dune_2021
dune_combined = join()("Dune Casts (combined)")
dune_combined.present(fields=["character", "actor_1984", "actor_2000", "actor_2021"])
