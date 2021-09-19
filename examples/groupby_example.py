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
1963,Best Actor,Gregory Peck,To Kill A Mockingbird
1963,Best Actress,Anne Bancroft,The Miracle Worker
1963,Best Director,David Lean,Lawrence of Arabia
1964,Best Picture,Tom Jones,
1964,Best Actor,Sidney Poitier,Lilies of the Field
1964,Best Actress,Patricia Neal,Hud
1964,Best Director,Tony Richardson,Tom Jones
1965,Best Picture,My Fair Lady,
1965,Best Actor,Rex Harrison,My Fair Lady
1965,Best Actress,Julie Andrews,Mary Poppins
1965,Best Director,George Kukor,My Fair Lady
1966,Best Picture,The Sound of Music,
1966,Best Actor,Lee Marvin,Cat Ballou
1966,Best Actress,Julie Christie,Darling
1966,Best Director,Robert Wise,The Sound of Music
1967,Best Picture,A Man for All Season,
1967,Best Actor,Paul Scofield,A Man for All Seasons
1967,Best Actress,Elizabeth Taylor,Who's Afraid of Virginia Woolf
1967,Best Director,Fred Zinnemann,A Man for All Seasons
1968,Best Picture,In The Heat of The Night,
1968,Best Actor,Rod Steiger,In The Heat of The Night
1968,Best Actress,Katherine Hepburn,Guess Who's Coming to Dinner
1968,Best Director,Mike Nichols,The Graduate
1969,Best Picture,Oliver!,
1969,Best Actor,Cliff Robertson,Charly
1969,Best Actress,Barbra Streisand,Funny Girl
1969,Best Director,Carol Reed,Oliver!
""")

tbl.present(groupby="year")
