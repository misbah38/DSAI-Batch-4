
import streamlit as st
import pandas as pd

st.title("My Dashboard")
st.write("This is a simple dashboard that I created with Streamlit and CSV data.")

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv("IMDB_Movies.csv")

# Create input fields and select boxes
search_input=st.text_input("Search a movie")

#create a list of filters
filters=["Movie_Name","Year_release","IMDB_Ratings","SexNudity","Violence","Profanity","KIM Rating"]

## Create a checkbox widget for each filter
checkbox = {} # <----------------------------------------- Zee : Initialize a dict to keep the status of each CB
for filter_name in filters:
    checkbox[filter_name] = st.sidebar.checkbox(filter_name) # <--- Zee : Put each CB in a dict 

# Filter the DataFrame based on the search input and the selected filters

filtered_df = df # <----- Zee : here you used df['title'], in DF there was no such column named Title

for filter_name in filters:
    if checkbox[filter_name] == True: # <---- Zee : Check if the current (in loop) CB is checked or not. 
        filtered_df = filtered_df[filtered_df[filter_name].str.contains(search_input, na=False)] # <--- Zee : na=False as the data is not cleaned

# Display the filtered DataFrame
st.dataframe(filtered_df)



