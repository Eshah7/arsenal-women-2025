import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

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
    
    defenders_df = df[df['Pos'].str.contains("DF", na=False)]
    defender_names = defenders_df['Player'].dropna().unique()
    selected_defender = st.selectbox("Select a Defender", defender_names)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1: 
        player_stats = df[df['Player'] == selected_defender].iloc[0]
        st.metric("Minutes Played", player_stats["Min"], border=True)
    
    with col2: 
        st.metric("Matches Played", player_stats["MP"], border=True)
    
    with col3: 
        st.metric("Matches Started", player_stats["Starts"], border=True)

    with col4: 
        st.metric("Number of 90s Played", player_stats["90s"], border=True)

st.header("The Decline of Montreal Canadiens", divider = "red")

st.write("Documenting the downfall of the French! <3")

st.subheader("Overall Metrics")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1: 
    st.metric("Number of Seasons Played", 114, "Since 1909" ,border=True)

with col2: 
    st.metric("Stanley Cups Won", 24, "#1 Team", border =True)

with col3: 
    st.metric("Last Stanley Cup Won", "1992-93", "-32 Years", border = True)

with col4: 
    st.metric("Number of Games Played", 7114, "Avg. 62 games", border = True)

with col5: 
    st.metric("Number of Reg. Games Won", 3596, "59% Win Rate", border = True)

with col6: 
    st.metric("Number of Reg. Games Lost", 2463, "-41% Loss Rate", border = True)

st.subheader("Win Rate Per Season Since 1992-93")

# Load the CSV
df2 = pd.read_csv("montreal canadiens data - combined data.csv")

# Only keep relevant columns
df2 = df2[['Season', 'Results']]

# Convert to win = 1, tie = 0.5, loss = 0
df2['Win_Value'] = df2['Results'].map({'W': 1, 'T': 0.5, 'L': 0})

# Group by season and compute win rate
win_rate_by_season = df2.groupby('Season')['Win_Value'].mean().reset_index()
win_rate_by_season.columns = ['Season', 'Win Rate']

# Create the line and point chart with red color
line = alt.Chart(win_rate_by_season).mark_line(color='red', strokeWidth=3).encode(
    x=alt.X('Season:N', sort=win_rate_by_season['Season'].tolist(), title='Season'),
    y=alt.Y('Win Rate:Q', scale=alt.Scale(domain=[0, 1]), title='Win Rate'),
    tooltip=['Season', 'Win Rate']
)

points = alt.Chart(win_rate_by_season).mark_point(color='red', size=60).encode(
    x='Season:N',
    y='Win Rate:Q',
    tooltip=['Season', 'Win Rate']
)

# Combine line and points
chart = (line + points).properties(
    width=700,
    height=400
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
)

# Display the chart
st.altair_chart(chart, use_container_width=True)

st.subheader("", divider="rainbow")

st.header("2024 International Women's Big Data Cup Analysis")

hockey_data = pd.read_csv('BDC_2024_Womens_Data.csv')

#Metric - Corsi
corsi_details = ['On Net', 'Missed', 'Blocked']
corsi_df = hockey_data[(hockey_data['Event'] == 'Shot') & (hockey_data['Detail 2'].isin(corsi_details))]
corsi_counts = corsi_df.groupby('Team').size().reset_index(name='Corsi For')

#Metric - Fenwick
fenwick_details = ['On Net', 'Missed'] 
fenwick_df = hockey_data[(hockey_data['Event'] == 'Shot') & (hockey_data['Detail 2'].isin(fenwick_details))]
fenwick_counts = fenwick_df.groupby('Team').size().reset_index(name='Fenwick For')

#XG?
# Step 1: Get all valid shots
valid_shots = hockey_data[(hockey_data['Event'] == 'Shot') & (hockey_data['Detail 2'].isin(['On Net', 'Missed', 'Blocked']))].copy()

# Step 2: Calculate distance to goal (assume net at (100, 42.5))
valid_shots['Shot_Distance'] = np.sqrt((100 - valid_shots['X Coordinate'])**2 + (42.5 - valid_shots['Y Coordinate'])**2)

# Step 3: Compute xG proxy
valid_shots['xG_proxy'] = 1 / (1 + valid_shots['Shot_Distance'])

# Step 4: Sum xG by team
xg_by_team = valid_shots.groupby('Team')['xG_proxy'].sum().reset_index(name='Total xG (proxy)')

# Step 5: Extract for Canada and USA
canada_xg = xg_by_team[xg_by_team['Team'] == 'Women - Canada']
usa_xg = xg_by_team[xg_by_team['Team'] == 'Women - United States']


st.subheader("Canada")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1: 
    canada_corsi = corsi_counts.loc[corsi_counts['Team'] == 'Women - Canada', 'Corsi For'].values[0]
    st.metric("Corsi Counts", canada_corsi,border=True)

with col2: 
    canada_fenwick = fenwick_counts.loc[fenwick_counts['Team'] == 'Women - Canada', 'Fenwick For'].values[0]
    st.metric("Fenwick Counts", canada_fenwick,border=True)

with col3: 
    canada_xg_value = float(canada_xg['Total xG (proxy)'].values[0])
    st.metric("Total xG", round(canada_xg_value, 4), border = True)

st.subheader("USA")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1: 
    usa_corsi = corsi_counts.loc[corsi_counts['Team'] == 'Women - United States', 'Corsi For'].values[0]
    st.metric("Corsi Counts", usa_corsi,border=True)

with col2: 
    usa_fenwick = fenwick_counts.loc[fenwick_counts['Team'] == 'Women - United States', 'Fenwick For'].values[0]
    st.metric("Fenwick Counts", usa_fenwick,border=True)

with col3: 
    usa_xg_value = float(usa_xg['Total xG (proxy)'].values[0])
    st.metric("Total xG", round(usa_xg_value, 4), border= True)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

st.title("Flipped Shot Density Heatmap Over Rink")

#load impage
rink_img = mpimg.imread("rink_coords.png")     

# Filter valid shots
shot_df = hockey_data[
    (hockey_data['Event'] == 'Shot') &
    (hockey_data['Detail 2'].isin(['On Net', 'Missed', 'Blocked']))
].copy()

# Optional: team filter
teams = shot_df['Team'].dropna().unique()
selected_team = st.selectbox("Select a team to display shots", ["All Teams"] + sorted(teams.tolist()))



if selected_team != "All Teams":
    shot_df = shot_df[shot_df['Team'] == selected_team]

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Flip the rink image horizontally
ax.imshow(rink_img, extent=[200, 0, 0, 85], aspect='auto', zorder=0)

# Heatmap of shot density
sns.kdeplot(
    data=shot_df,
    x='X Coordinate',
    y='Y Coordinate',
    cmap='coolwarm',
    fill=True,
    thresh=0.01,
    alpha=0.7,
    bw_adjust=1.2,
    ax=ax,
    zorder=1
)

# Format plot
ax.set_xlim(200, 0)
ax.set_ylim(0, 85)
ax.set_title(f"Shot Density Heatmap - {selected_team}", fontsize=14, color='white' if st.get_option('theme.base') == 'dark' else 'black')
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")

# Optional: dark theme support
ax.set_facecolor('black' if st.get_option('theme.base') == 'dark' else 'white')
fig.patch.set_facecolor('black' if st.get_option('theme.base') == 'dark' else 'white')

# Display
st.pyplot(fig)