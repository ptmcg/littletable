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
recipe_data = textwrap.dedent(
    """\
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
    """
)
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
    if matches:
        for rec in matches:
            print(" -", rec.title, rec.ingredients_search_score)
    else:
        print("  <no matching recipes>")
    print()

# redo last match, getting the words for each match
print("repeat last match, including matched words")
matches = recipes.search.ingredients(
    query, limit=5, min_score=-100000, include_words=True
)
print(query)
for rec in matches:
    print(" -", rec.title,rec.ingredients_search_score, rec.ingredients_search_words)

# build a more complex search index, using words from multiple fields
recipes.create_search_index("recipe_terms", using="ingredients title")
matches = recipes.search.recipe_terms("casserole Hawaiian", as_table=True)
matches.present(fields="title ingredients".split())

# exception gets raised if search() is called after the table has been modified without
# rebuilding the search index
print("\nmodify source table to show exception raised when search index is no longer valid")
recipes.pop(0)
matches = recipes.search.ingredients(query)
