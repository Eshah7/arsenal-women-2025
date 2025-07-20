import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="WNBA Draft Success Analysis",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🏀 WNBA Draft Success Analysis")
st.markdown("""
This interactive analysis explores how WNBA draft position correlates with career outcomes.
Upload your WNBA draft dataset to begin the analysis.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your WNBA draft CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        return df
    
    df = load_data(uploaded_file)
    
    # Data preprocessing
    @st.cache_data
    def preprocess_data(df):
        # Create a copy for processing
        df_processed = df.copy()
        
        # Fill missing values for key columns
        numeric_cols = ['games', 'win_shares', 'win_shares_40', 'minutes_played', 'points', 'total_rebounds', 'assists']
        for col in numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(0)
        
        # Create derived features
        df_processed['played_in_wnba'] = (df_processed['years_played'] > 0) & (df_processed['games'] > 0)
        df_processed['has_college'] = df_processed['college'].notna()
        df_processed['is_international'] = df_processed['former'].notna()
        
        # Create draft round (assuming 12 picks per round)
        df_processed['draft_round'] = ((df_processed['overall_pick'] - 1) // 12) + 1
        
        # Create pick ranges
        df_processed['pick_range'] = pd.cut(df_processed['overall_pick'], 
                                          bins=[0, 5, 10, 15, 20, 30, float('inf')], 
                                          labels=['1-5', '6-10', '11-15', '16-20', '21-30', '30+'])
        
        # Create success metrics
        df_processed['long_career'] = df_processed['years_played'] >= 3
        df_processed['high_impact'] = df_processed['win_shares'] > df_processed['win_shares'].median()
        
        return df_processed
    
    df_processed = preprocess_data(df)
    
    # Sidebar for analysis options
    st.sidebar.header("Analysis Options")
    
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Overview", "Draft Position Analysis", "Career Success Metrics", "Predictive Modeling", "Team Analysis"]
    )
    
    # Main analysis sections
    if analysis_type == "Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Players", len(df_processed))
        with col2:
            st.metric("Players Who Played", df_processed['played_in_wnba'].sum())
        with col3:
            st.metric("Success Rate", f"{df_processed['played_in_wnba'].mean():.1%}")
        with col4:
            st.metric("Years Covered", f"{df_processed['year'].min()}-{df_processed['year'].max()}")
        
        # Data quality summary
        st.subheader("Data Quality Summary")
        
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df) * 100).round(2)
        
        quality_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_pct.values
        }).sort_values('Missing %', ascending=False)
        
        st.dataframe(quality_df)
        
        # Sample data
        st.subheader("Sample Data")
        st.dataframe(df_processed.head(10))
    
    elif analysis_type == "Draft Position Analysis":
        st.header("🎯 Draft Position vs Career Outcomes")
        
        # Key question: At what pick do players usually get playing time?
        st.subheader("Playing Time by Draft Position")
        
        # Create aggregated data by pick
        pick_analysis = df_processed.groupby('overall_pick').agg({
            'played_in_wnba': ['count', 'sum', 'mean'],
            'years_played': 'mean',
            'games': 'mean',
            'win_shares': 'mean'
        }).round(2)
        
        pick_analysis.columns = ['Total_Drafted', 'Played_Count', 'Play_Rate', 'Avg_Years', 'Avg_Games', 'Avg_WinShares']
        pick_analysis = pick_analysis.reset_index()
        
        # Plot 1: Probability of playing by pick
        fig1 = px.line(pick_analysis, x='overall_pick', y='Play_Rate',
                      title='Probability of Playing in WNBA by Draft Pick',
                      labels={'Play_Rate': 'Probability of Playing', 'overall_pick': 'Draft Pick'})
        fig1.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="50% threshold")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Plot 2: Average career length by pick range
        range_analysis = df_processed.groupby('pick_range').agg({
            'played_in_wnba': ['count', 'sum', 'mean'],
            'years_played': 'mean',
            'games': 'mean',
            'win_shares': 'mean'
        }).round(2)
        
        range_analysis.columns = ['Total', 'Played', 'Play_Rate', 'Avg_Years', 'Avg_Games', 'Avg_WinShares']
        range_analysis = range_analysis.reset_index()
        
        fig2 = px.bar(range_analysis, x='pick_range', y='Play_Rate',
                     title='Success Rate by Draft Pick Range',
                     labels={'Play_Rate': 'Probability of Playing', 'pick_range': 'Pick Range'})
        st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed analysis table
        st.subheader("Detailed Analysis by Pick Range")
        st.dataframe(range_analysis)
        
        # Key insights
        st.subheader("🔍 Key Insights")
        
        # Find the pick where success rate drops below 50%
        threshold_pick = pick_analysis[pick_analysis['Play_Rate'] < 0.5]['overall_pick'].min()
        
        st.write(f"• **Critical Pick Threshold**: Success rate drops below 50% around pick #{threshold_pick}")
        st.write(f"• **Top 5 Picks**: {range_analysis.loc[0, 'Play_Rate']:.1%} success rate")
        st.write(f"• **Late Picks (30+)**: {range_analysis.loc[range_analysis['pick_range'] == '30+', 'Play_Rate'].iloc[0]:.1%} success rate")
    
    elif analysis_type == "Career Success Metrics":
        st.header("📈 Career Success Analysis")
        
        # Multiple success metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Years played distribution
            fig3 = px.histogram(df_processed[df_processed['played_in_wnba']], 
                              x='years_played', 
                              title='Distribution of Career Length (Years Played)',
                              nbins=20)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Win shares distribution
            fig4 = px.histogram(df_processed[df_processed['win_shares'] > 0], 
                              x='win_shares', 
                              title='Distribution of Career Win Shares',
                              nbins=30)
            st.plotly_chart(fig4, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Career Metrics by Draft Position")
        
        # Create scatter plot of pick vs various metrics
        metric_choice = st.selectbox(
            "Choose metric to analyze:",
            ['years_played', 'games', 'win_shares', 'minutes_played', 'points']
        )
        
        fig5 = px.scatter(df_processed[df_processed[metric_choice] > 0], 
                         x='overall_pick', 
                         y=metric_choice,
                         color='pick_range',
                         title=f'{metric_choice.title()} by Draft Pick',
                         trendline="ols")
        st.plotly_chart(fig5, use_container_width=True)
        
        # Top performers analysis
        st.subheader("Top Performers by Draft Range")
        
        top_performers = df_processed[df_processed['win_shares'] > 0].nlargest(20, 'win_shares')[
            ['player', 'overall_pick', 'pick_range', 'years_played', 'win_shares', 'team']
        ]
        st.dataframe(top_performers)
    
    elif analysis_type == "Predictive Modeling":
        st.header("🤖 Predictive Modeling")
        
        st.subheader("Predict: Will a Player Play in the WNBA?")
        
        # Prepare features for modeling
        features_for_modeling = ['overall_pick', 'has_college', 'is_international', 'draft_round']
        
        # Handle categorical variables
        df_model = df_processed.copy()
        
        # Encode team (optional - might have too many categories)
        if st.checkbox("Include drafting team as a feature"):
            le_team = LabelEncoder()
            df_model['team_encoded'] = le_team.fit_transform(df_model['team'])
            features_for_modeling.append('team_encoded')
        
        # Prepare data
        X = df_model[features_for_modeling].copy()
        y = df_model['played_in_wnba'].copy()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model selection
        model_type = st.selectbox("Choose Model:", ["Random Forest", "Logistic Regression"])
        
        if st.button("Train Model"):
            if model_type == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=42)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Accuracy", f"{accuracy:.2%}")
                
                # Feature importance (if Random Forest)
                if model_type == "Random Forest":
                    feature_importance = pd.DataFrame({
                        'feature': features_for_modeling,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig6 = px.bar(feature_importance, x='importance', y='feature',
                                 title='Feature Importance', orientation='h')
                    st.plotly_chart(fig6, use_container_width=True)
            
            with col2:
                # Classification report
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))
        
        # Interactive prediction
        st.subheader("Make a Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_pick = st.number_input("Overall Pick", min_value=1, max_value=50, value=10)
        with col2:
            pred_college = st.selectbox("Has College?", [True, False])
        with col3:
            pred_international = st.selectbox("International Player?", [True, False])
        
        pred_round = ((pred_pick - 1) // 12) + 1
        
        # This would require the trained model to be persistent
        st.write(f"Draft Round: {pred_round}")
        st.write("Note: Train the model above to make predictions")
    
    elif analysis_type == "Team Analysis":
        st.header("🏀 Team Drafting Analysis")
        
        st.subheader("Which Teams Have Had the Most Draft Success?")
        
        # Team success metrics
        team_analysis = df_processed.groupby('team').agg({
            'played_in_wnba': ['count', 'sum', 'mean'],
            'years_played': 'mean',
            'win_shares': 'sum',
            'games': 'mean'
        }).round(2)
        
        team_analysis.columns = ['Total_Drafted', 'Played_Count', 'Success_Rate', 'Avg_Career_Length', 'Total_WinShares', 'Avg_Games']
        team_analysis = team_analysis.reset_index()
        team_analysis = team_analysis.sort_values('Success_Rate', ascending=False)
        
        # Plot team success rates
        fig7 = px.bar(team_analysis.head(15), x='team', y='Success_Rate',
                     title='Team Draft Success Rate (Top 15 Teams)',
                     labels={'Success_Rate': 'Success Rate', 'team': 'Team'})
        fig7.update_xaxes(tickangle=45)
        st.plotly_chart(fig7, use_container_width=True)
        
        # Detailed team table
        st.subheader("Complete Team Analysis")
        st.dataframe(team_analysis)
        
        # Team-specific analysis
        st.subheader("Analyze Specific Team")
        selected_team = st.selectbox("Choose a team:", df_processed['team'].unique())
        
        team_data = df_processed[df_processed['team'] == selected_team]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Drafted", len(team_data))
        with col2:
            st.metric("Players Who Played", team_data['played_in_wnba'].sum())
        with col3:
            st.metric("Success Rate", f"{team_data['played_in_wnba'].mean():.1%}")
        
        # Team's draft picks over time
        team_yearly = team_data.groupby('year').agg({
            'played_in_wnba': ['count', 'sum'],
            'overall_pick': 'mean'
        })
        team_yearly.columns = ['Picks', 'Successful', 'Avg_Pick']
        team_yearly = team_yearly.reset_index()
        
        if len(team_yearly) > 1:
            fig8 = px.line(team_yearly, x='year', y='Successful',
                          title=f'{selected_team} - Successful Picks Over Time')
            st.plotly_chart(fig8, use_container_width=True)

else:
    st.info("Please upload your WNBA draft CSV file to begin the analysis.")
    
    # Show expected data format
    st.subheader("Expected Data Format")
    st.markdown("""
    Your CSV should contain the following columns:
    - `overall_pick`: Draft pick number
    - `year`: Draft year
    - `team`: Drafting team
    - `player`: Player name
    - `college`: College attended (if applicable)
    - `former`: Former team/league (if applicable)
    - `years_played`: Years played in WNBA
    - `games`: Total games played
    - `win_shares`: Career win shares
    - `minutes_played`: Average minutes per game
    - `points`: Average points per game
    - `total_rebounds`: Average rebounds per game
    - `assists`: Average assists per game
    """)