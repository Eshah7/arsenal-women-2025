import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Arsenal Women Optimal Line Ups",
    page_icon="⚽",
    layout="wide"
)

st.title("Determining Arsenal Women's Optimal Player Lineup Against Different Opponent Styles")
##st.markdown('')
##st.markdown("#### Opinion By: Esha Shah")
st.subheader("", divider="rainbow")
st.markdown('')
st.subheader("Opinion Goal")

st.write("""Although Arsenal and the WSL in general is known to have a more of a build-up style of play, certain players of the team 
         bring diversity in the style Arsenal forms its attack throughout the 2024-2025 season."""
         )
st.write("""How did Arsenal perform against the different types of styles 
         and what would the optimal line up be for each style? For this particular analysis, two types of styles will be defined.
         Thereafter, each players statistics against the different styles of play will help us determine which style of play they are better suited for through a decision 
         decision support system. """)

st.subheader("Defining the Types of Styles")
st.write("The two types of styles that will be build-up style of play and a fast direct style.")

col1, col2 = st.columns(2)
with col1: 
    st.image("images/build-up-style.png", width=900)
    st.markdown(":green-background[Build Up Style]")

with col2: 
    st.image("images/direct-style.png", width=900)
    st.markdown(":green-background[Fast Direct Style]")

st.write("In a build up style of play, players will often complete an X amount of progression passes and maximize their posession before shooting towards the goal.")

st.write("""While in a fast direct style of play, players will often progress across the field and shoot towards the target while minmizing the number of passes and time.
        This style leads to more turnovers and transitions between the two teams.""")

st.subheader("Player Level Analysis")
st.write("2024-2025 Season Statistics for All Games")


df = pd.read_csv("data/Arsenal Women Data (Snapshot_ Jun 4th 2025) - Team Level_ All Competitions.csv")
columns = st.multiselect("Select Columns to View:", df.columns.tolist(), default=df.columns[:17])
st.dataframe(df[columns], height=300)

tab1, tab2, tab3, tab4 = st.tabs(["Defenders", "Midfielders", "Forwards", "Goalkeepers"])

with tab1: 
    st.markdown("#### Defenders")

    col1, col2, col3 = st.columns(3)
    with col1: 
         st.markdown(
        ":red-background[Emily Fox]"
         )


