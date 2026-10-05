import streamlit as st

# Define the pages
main_page = st.Page("pages/home.py", title="Main Page", icon="🏠")
coffee_classify = st.Page(
    "pages/coffee_classification_app.py",
    title="Coffee Classification",
    icon="☕"
)

# Set up navigation
pg = st.navigation([main_page, coffee_classify])

# Run the selected page
pg.run()