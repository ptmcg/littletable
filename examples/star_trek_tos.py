import littletable as lt

st = lt.Table().csv_import(
    "star_trek_tos_eps.csv",
    transforms={"rating": float, "votes": int}
)

# sort by rating, and add "rank" field
st.sort(["rating desc", "votes desc"])
for i, rec in enumerate(st, start=1):
    setattr(rec, "rank", i)

# display 10 best and worst episodes
fields = "rank date title rating votes".split()
count = 10
best_and_worst = (st[:count] + st[-count:])(f"{count} Best and Worst Star Trek Episodes")
best_and_worst.present(
    fields,
    caption="data from IMDB.com",
    caption_justify="right"
)

# full-text search of descriptions
st.create_search_index("description")
for query in [
    "--kirk ++spock",
    "--kirk ++mccoy",
]:
    st.search.description(query, as_table=True).select("date title description").present()
