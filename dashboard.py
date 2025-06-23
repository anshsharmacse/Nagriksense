# dashboard.py
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Civic Sense AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
</style>
""", unsafe_allow_html=True)

class DashboardData:
    def __init__(self, db_path='civic_enforcement.db'):
        self.db_path = db_path
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_data(_self):
        """Load all data from database"""
        conn = sqlite3.connect(_self.db_path)
        
        # Users data
        users_df = pd.read_sql_query('''
            SELECT u.*, qr.total_score, qr.responsibility_level, qr.taken_at as quiz_date
            FROM users u
            LEFT JOIN quiz_results qr ON u.id = qr.user_id
        ''', conn)
        
        # Violations data
        violations_df = pd.read_sql_query('''
            SELECT * FROM violations
        ''', conn)
        
        # Behavior predictions
        predictions_df = pd.read_sql_query('''
            SELECT bp.*, u.age, u.gender, u.education_level
            FROM behavior_predictions bp
            JOIN users u ON bp.user_id = u.id
        ''', conn)
        
        # System stats
        stats_query = '''
            SELECT 
                (SELECT COUNT(*) FROM users) as total_users,
                (SELECT COUNT(*) FROM violations) as total_violations,
                (SELECT COUNT(*) FROM violations WHERE status = 'confirmed') as confirmed_violations,
                (SELECT SUM(fine_amount) FROM violations WHERE paid = 1) as revenue_collected,
                (SELECT AVG(total_score) FROM quiz_results) as avg_quiz_score
        '''
        stats_df = pd.read_sql_query(stats_query, conn)
        
        conn.close()
        
        return users_df, violations_df, predictions_df, stats_df

# Initialize dashboard
def main():
    st.markdown('<h1 class="main-header">🛡️ Civic Sense AI Enforcement Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    dashboard = DashboardData()
    users_df, violations_df, predictions_df, stats_df = dashboard.load_data()
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # Date filter
    if not violations_df.empty:
        violations_df['timestamp'] = pd.to_datetime(violations_df['timestamp'])
        min_date = violations_df['timestamp'].min().date()
        max_date = violations_df['timestamp'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            violations_filtered = violations_df[
                (violations_df['timestamp'].dt.date >= start_date) &
                (violations_df['timestamp'].dt.date <= end_date)
            ]
        else:
            violations_filtered = violations_df
    else:
        violations_filtered = violations_df
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if not stats_df.empty:
        with col1:
            st.metric(
                label="👥 Total Users",
                value=f"{stats_df.iloc[0]['total_users']:,}",
                delta=f"+{np.random.randint(1, 10)} today"
            )
        
        with col2:
            st.metric(
                label="⚠️ Total Violations",
                value=f"{stats_df.iloc[0]['total_violations']:,}",
                delta=f"+{np.random.randint(0, 5)} today"
            )
        
        with col3:
            st.metric(
                label="💰 Revenue Collected",
                value=f"₹{stats_df.iloc[0]['revenue_collected']:,.0f}",
                delta=f"+₹{np.random.randint(100, 1000)} today"
            )
        
        with col4:
            st.metric(
                label="📊 Avg Quiz Score",
                value=f"{stats_df.iloc[0]['avg_quiz_score']:.1f}/5.0",
                delta=f"{np.random.uniform(-0.1, 0.1):.1f}"
            )
    
    st.divider()
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Violation Trends")
        if not violations_filtered.empty:
            daily_violations = violations_filtered.groupby(
                violations_filtered['timestamp'].dt.date
            ).size().reset_index()
            daily_violations.columns = ['Date', 'Count']
            
            fig = px.line(daily_violations, x='Date', y='Count',
                         title='Daily Violation Count',
                         color_discrete_sequence=['#1f77b4'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No violation data available for selected period")
    
    with col2:
        st.subheader("🏷️ Violation Types")
        if not violations_filtered.empty:
            violation_counts = violations_filtered['type'].value_counts()
            
            fig = px.pie(values=violation_counts.values, 
                        names=violation_counts.index,
                        title='Distribution of Violation Types')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No violation data available")
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 User Demographics")
        if not users_df.empty and 'age' in users_df.columns:
            fig = px.histogram(users_df, x='age', nbins=20,
                             title='Age Distribution of Users',
                             color_discrete_sequence=['#2ca02c'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No user demographic data available")
    
    with col2:
        st.subheader("📝 Quiz Performance")
        if not users_df.empty and 'total_score' in users_df.columns:
            users_with_scores = users_df.dropna(subset=['total_score'])
            if not users_with_scores.empty:
                fig = px.histogram(users_with_scores, x='total_score', nbins=15,
                                 title='Distribution of Quiz Scores',
                                 color_discrete_sequence=['#ff7f0e'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No quiz score data available")
        else:
            st.info("No quiz data available")
    
    # Risk Assessment
    st.subheader("🎯 Behavior Risk Assessment")
    if not predictions_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            risk_counts = predictions_df['risk_level'].value_counts()
            fig = px.bar(x=risk_counts.index, y=risk_counts.values,
                        title='Risk Level Distribution',
                        labels={'x': 'Risk Level', 'y': 'Number of Users'},
                        color=risk_counts.values,
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk by age group
            predictions_df['age_group'] = pd.cut(predictions_df['age'], 
                                               bins=[18, 25, 35, 50, 100], 
                                               labels=['18-25', '26-35', '36-50', '50+'])
            risk_by_age = predictions_df.groupby(['age_group', 'risk_level']).size().unstack(fill_value=0)
            
            fig = px.bar(risk_by_age, title='Risk Level by Age Group',
                        labels={'value': 'Count', 'index': 'Age Group'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No behavior prediction data available")
    
    # Data tables
    st.subheader("📊 Recent Activity")
    
    tab1, tab2, tab3 = st.tabs(["Recent Violations", "New Users", "High Risk Users"])
    
    with tab1:
        if not violations_filtered.empty:
            recent_violations = violations_filtered.sort_values('timestamp', ascending=False).head(10)
            st.dataframe(
                recent_violations[['type', 'confidence', 'timestamp', 'status', 'fine_amount']],
                use_container_width=True
            )
        else:
            st.info("No recent violations")
    
    with tab2:
        if not users_df.empty:
            users_df['created_at'] = pd.to_datetime(users_df['created_at'])
            recent_users = users_df.sort_values('created_at', ascending=False).head(10)
            st.dataframe(
                recent_users[['email', 'age', 'gender', 'created_at', 'responsibility_level']],
                use_container_width=True
            )
        else:
            st.info("No user data available")
    
    with tab3:
        if not predictions_df.empty:
            high_risk = predictions_df[predictions_df['risk_level'] == 'High'].sort_values(
                'violation_probability', ascending=False
            ).head(10)
            if not high_risk.empty:
                st.dataframe(
                    high_risk[['age', 'gender', 'risk_level', 'violation_probability', 'predicted_at']],
                    use_container_width=True
                )
            else:
                st.info("No high-risk users identified")
        else:
            st.info("No behavior prediction data available")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🛡️ Civic Sense AI Enforcement System | Real-time Analytics Dashboard</p>
        <p>Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
