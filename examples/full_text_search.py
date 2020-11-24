#
# full_text_search.py
#
# Example code using littletable with a search index to do full text searching
# of a table attribute.
#
# Copyright (c) 2020  Paul T. McGuire
#

import littletable as lt
import textwrap

# create table of recipes, by title and ingredients
recipe_data = textwrap.dedent("""\
    title,ingredients
    Tuna casserole,tuna noodles cream of mushroom soup
    Hawaiian pizza,pizza dough pineapple ham tomato sauce
    Margherita pizza,pizza dough cheese pesto artichoke hearts
    Pepperoni pizza,pizza dough cheese tomato sauce pepperoni
    Grilled cheese sandwich,bread cheese butter
    Tuna melt,tuna mayonnaise tomato bread cheese
    Chili dog,hot dog chili onion bun
    French toast,egg milk vanilla bread maple syrup
    BLT,bread bacon lettuce tomato mayonnaise
    Reuben sandwich,rye bread sauerkraut corned beef swiss cheese russian dressing thousand island
    Hamburger,ground beef bun lettuce ketchup mustard pickle
    Cheeseburger,ground beef bun lettuce ketchup mustard pickle cheese
    Bacon cheeseburger,ground beef bun lettuce ketchup mustard pickle cheese bacon
    """)
recipes = lt.Table().csv_import(recipe_data)

# define search index on "ingredients" attribute
search_attr = "ingredients"
recipes.create_search_index(search_attr)

# run sample queries
queries = """\
    tuna
    tuna +cheese
    pineapple +bacon lettuce beef -sauerkraut tomato
    pizza dough -pineapple
    pizza dough --pineapple
    bread bacon
    bread ++bacon""".splitlines()

# run each query, listing top 5 matches
for query in queries:
    query = query.strip()
    if not query:
        continue

    matches = recipes.search.ingredients(query, limit=5, min_score=-100000)

    print(query)
    for rec, score in matches:
        print(" -", rec.title, score)
    print()

# redo last match, getting the words for each match
matches = recipes.search.ingredients(query, limit=5, min_score=-100000, include_words=True)
print(query)
for rec, score, search_words in matches:
    print(" -", rec.title, score, search_words)

# exception gets raised if search() is called after the table has been modified without
# rebuilding the search index
recipes.pop(0)
matches = recipes.search.ingredients(query)
