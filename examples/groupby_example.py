import littletable as lt

tbl = lt.Table("Academy Awards 1960-1969").csv_import("""\
year,award,movie,recipient
1960,Best Picture,Ben-Hur,
1960,Best Actor,Ben-Hur,Charlton Heston
1960,Best Actress,The Heiress,Simone Signoret
1960,Best Director,Ben-Hur,William Wyler
1961,Best Picture,The Apartment,
1961,Best Actor,Elmer Gantry,Burt Lancaster
1961,Best Actress,Butterfield 8,Elizabeth Taylor
1961,Best Director,The Apartment,Billy Wilder
1962,Best Picture,West Side Story,
1962,Best Actor,Judgment at Nuremberg,Maximilian Schell
1962,Best Actress,Two Women,Sophia Loren
1962,Best Director,West Side Story,Willian Wise/Jerome Robbins
1963,Best Picture,Lawrence of Arabia,
1963,Best Actor,To Kill A Mockingbird,Gregory Peck
1963,Best Actress,The Miracle Worker,Anne Bancroft
1963,Best Director,Lawrence of Arabia,David Lean
1964,Best Picture,Tom Jones,
1964,Best Actor,Lilies of the Field,Sidney Poitier
1964,Best Actress,Hud,Patricia Neal
1964,Best Director,Tom Jones,Tony Richardson
1965,Best Picture,My Fair Lady,
1965,Best Actor,My Fair Lady,Rex Harrison
1965,Best Actress,Mary Poppins,Julie Andrews
1965,Best Director,My Fair Lady,George Kukor
1966,Best Picture,The Sound of Music,
1966,Best Actor,Cat Ballou,Lee Marvin
1966,Best Actress,Darling,Julie Christie
1966,Best Director,The Sound of Music,Robert Wise
1967,Best Picture,A Man for All Season,
1967,Best Actor,A Man for All Seasons,Paul Scofield
1967,Best Actress,Who's Afraid of Virginia Woolf,Elizabeth Taylor
1967,Best Director,A Man for All Seasons,Fred Zinnemann
1968,Best Picture,In The Heat of The Night,
1968,Best Actor,In The Heat of The Night,Rod Steiger
1968,Best Actress,Guess Who's Coming to Dinner,Katherine Hepburn
1968,Best Director,The Graduate,Mike Nichols
1969,Best Picture,Oliver!,
1969,Best Actor,Charly,Cliff Robertson
1969,Best Actress,Funny Girl,Barbra Streisand
1969,Best Director,Oliver!,Carol Reed
""")

tbl.present(groupby="year")
