import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import re
import json
import google.generativeai as genai
from itertools import combinations
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Diving CRM Dashboard",
    page_icon="🤿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
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
        margin: 0.5rem 0;
    }
    .customer-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f9f9f9;
    }
    .ai-response {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .rule-card {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sample data generation (replace with your CSV loading)
@st.cache_data
def load_sample_data():
    """Load and prepare sample data based on your CSV structure"""
    try:
        data = pd.read_csv(r"ssi customers.csv", encoding='utf-8')
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(r"ssi customers.csv", encoding='latin-1')
        except UnicodeDecodeError:
            data = pd.read_csv(r"ssi customers.csv", encoding='cp1252')
    
    df = pd.DataFrame(data)
    # Convert Dives column to numeric, handling non-numeric values
    df['Dives'] = pd.to_numeric(df['Dives'], errors='coerce').fillna(0)
    
    # Convert Specialties column to numeric, handling non-numeric values
    df['Specialties'] = pd.to_numeric(df['Specialties'], errors='coerce').fillna(0)
    
    # Add some derived columns for better analytics
    df['Total_Courses'] = df['Courses'].apply(lambda x: len(x.split(',')) if pd.notna(x) else 0)
    df['Has_Phone'] = df['Phone'].apply(lambda x: 1 if x and str(x) not in ['0', '', 'nan'] else 0)
    df['Has_Mobile'] = df['Mobile'].apply(lambda x: 1 if x and str(x) not in ['0', '', 'nan'] else 0)
    df['Customer_Type'] = df['Dives'].apply(lambda x: 'Advanced' if x > 100 else 'Intermediate' if x > 20 else 'Beginner')
    df['Full_Name'] = df['First Name'] + ' ' + df['Last Name']
    
    return df

# Enhanced AI Assistant with Full Data Analysis
class SmartDivingCRMAI:
    def __init__(self, data, gemini_api_key=None):
        self.data = data
        self.gemini_api_key = gemini_api_key
        
        # Initialize Gemini if API key is provided
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                self.use_gemini = True
            except Exception as e:
                st.error(f"Failed to initialize Gemini: {str(e)}")
                self.use_gemini = False
        else:
            self.use_gemini = False
    
    def get_data_summary(self):
        """Get comprehensive data summary for AI context"""
        summary = {
            "total_customers": len(self.data),
            "total_dives": int(self.data['Dives'].sum()),
            "average_dives": float(self.data['Dives'].mean()),
            "customer_types": self.data['Customer_Type'].value_counts().to_dict(),
            "top_courses": self.get_top_courses(10),
            "email_completion": float((self.data['Email'].notna()).mean() * 100),
            "phone_completion": float(self.data['Has_Phone'].mean() * 100),
            "mobile_completion": float(self.data['Has_Mobile'].mean() * 100),
            "specialties_distribution": self.data['Specialties'].value_counts().to_dict(),
            "dive_ranges": {
                "0_dives": int((self.data['Dives'] == 0).sum()),
                "1_20_dives": int(((self.data['Dives'] > 0) & (self.data['Dives'] <= 20)).sum()),
                "21_100_dives": int(((self.data['Dives'] > 20) & (self.data['Dives'] <= 100)).sum()),
                "100plus_dives": int((self.data['Dives'] > 100).sum())
            }
        }
        return summary
    
    def get_top_courses(self, n=10):
        """Get top courses with counts"""
        all_courses = []
        for courses in self.data['Courses'].dropna():
            course_list = [course.strip() for course in courses.split(',')]
            all_courses.extend(course_list)
        
        course_counts = pd.Series(all_courses).value_counts().head(n)
        return course_counts.to_dict()
    
    def create_comprehensive_prompt(self, user_query):
        """Create a comprehensive prompt for full data analysis"""
        
        # Get data summary
        data_summary = self.get_data_summary()
        
        # Get sample records for context
        sample_data = self.data.head(3).to_dict('records')
        
        examples_block = """
EXAMPLES (strictly illustrative, adapt values to the user's query):
1) Query: "Show customers who completed Try Scuba"
```json
{
  "query_type": "filter",
  "filters": [
    {"column": "Courses", "type": "contains", "value": "try scuba", "case_sensitive": false}
  ],
  "response": "Customers who have 'Try Scuba' among their completed courses"
}
```

2) Query: "Only Advanced divers with more than 50 dives"
```json
{
  "query_type": "filter",
  "filters": [
    {"column": "Customer_Type", "value": "Advanced"},
    {"column": "Dives", "operator": ">", "value": 50}
  ],
  "response": "Advanced customers with dive count greater than 50"
}
```
"""

        prompt = f"""
You are an advanced AI assistant for a diving center CRM system. You have access to the complete customer database and can answer ANY question about the data.

COMPREHENSIVE DATA OVERVIEW:
{json.dumps(data_summary, indent=2)}

SAMPLE CUSTOMER RECORDS:
{json.dumps(sample_data, indent=2)}

DATA SCHEMA:
- SSI_Master_ID: Unique customer identifier
- First_Name, Last_Name, Full_Name: Customer names
- Email, Phone, Mobile: Contact information
- Dives: Total number of dives completed
- Specialties: Number of specialty certifications
- Courses: Comma-separated list of courses taken
- Total_Courses: Count of courses completed
- Customer_Type: Beginner (≤20 dives), Intermediate (21-100), Advanced (>100)
- Has_Phone, Has_Mobile: Boolean indicators for contact completeness

USER QUERY: "{user_query}"

INSTRUCTIONS:
1. Analyze the user's question thoroughly
2. If it's a data filtering/search query, provide JSON filtering instructions
3. If it's an analytical question, provide insights based on the data summary
4. If it's asking for recommendations or business insights, use your knowledge of diving industry best practices
5. Always be specific and actionable in your responses

For filtering queries, return JSON with:
- "query_type": "filter" or "analysis" or "recommendation"
- "filters": Array of filter objects (if applicable)
- "analysis_type": Type of analysis requested
- "response": Your detailed response to the user's question

For analysis queries, provide:
- Key insights
- Relevant statistics
- Business recommendations
- Actionable next steps

RESPOND WITH VALID JSON ONLY.
{examples_block}
"""
        return prompt
    
    def execute_smart_query(self, user_query):
        """Execute comprehensive query analysis"""
        if not self.use_gemini:
            return self.fallback_analysis(user_query)
        
        try:
            prompt = self.create_comprehensive_prompt(user_query)
            response = self.model.generate_content(prompt)
            
            # Parse the response
            result = self.parse_gemini_response(response.text)
            
            if not result:
                return self.fallback_analysis(user_query)
            
            # Handle different query types
            if result.get('query_type') == 'filter':
                filters_from_ai = result.get('filters', [])
                filtered_data = self.apply_filters(filters_from_ai)

                # Guardrail: if AI produced no effective filters (returns all rows),
                # fall back to deterministic heuristic parsing to avoid showing everything.
                if len(self.data) == len(filtered_data) or not filters_from_ai:
                    heuristic_data, parsed = self.fallback_execute_query(user_query)
                    if len(heuristic_data) != len(self.data):
                        filtered_data = heuristic_data
                        # Enrich response to indicate heuristic was applied
                        result['response'] = result.get('response', 'Data filtered') + " (applied heuristic filters)"
                        result['interpretation'] = str(parsed)

                return {
                    'type': 'filter',
                    'data': filtered_data,
                    'response': result.get('response', 'Data filtered successfully'),
                    'interpretation': result.get('interpretation', result.get('response', ''))
                }
            elif result.get('query_type') == 'analysis':
                return {
                    'type': 'analysis',
                    'response': result.get('response', 'Analysis completed'),
                    'data_summary': self.get_data_summary(),
                    'analysis_type': result.get('analysis_type', 'general')
                }
            else:
                return {
                    'type': 'recommendation',
                    'response': result.get('response', 'Recommendations provided'),
                    'data_summary': self.get_data_summary()
                }
                
        except Exception as e:
            st.error(f"Error with AI analysis: {str(e)}")
            return self.fallback_analysis(user_query)
    
    def parse_gemini_response(self, response_text):
        """Parse Gemini's JSON response"""
        try:
            clean_response = response_text.strip()
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:-3]
            elif clean_response.startswith('```'):
                clean_response = clean_response[3:-3]
            
            return json.loads(clean_response)
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse AI response: {e}")
            return None
    
    def apply_filters(self, filters_list):
        """Apply filters to the data"""
        filtered_data = self.data.copy()
        
        for filter_config in filters_list:
            if 'column' not in filter_config:
                continue
                
            column = filter_config['column']
            if column not in filtered_data.columns:
                continue
            
            # Apply different filter types
            if 'operator' in filter_config:  # Numeric filter
                operator = filter_config['operator']
                value = filter_config['value']
                
                if operator == '>':
                    filtered_data = filtered_data[filtered_data[column] > value]
                elif operator == '>=':
                    filtered_data = filtered_data[filtered_data[column] >= value]
                elif operator == '<':
                    filtered_data = filtered_data[filtered_data[column] < value]
                elif operator == '<=':
                    filtered_data = filtered_data[filtered_data[column] <= value]
                elif operator == '=':
                    filtered_data = filtered_data[filtered_data[column] == value]
                elif operator == '!=':
                    filtered_data = filtered_data[filtered_data[column] != value]
                    
            elif 'type' in filter_config and filter_config['type'] == 'contains':  # Text filter
                value = filter_config['value']
                case_sensitive = filter_config.get('case_sensitive', False)
                filtered_data = filtered_data[
                    filtered_data[column].astype(str).str.contains(
                        value, case=case_sensitive, na=False
                    )
                ]
                
            elif 'value' in filter_config and 'type' not in filter_config:  # Category filter
                value = filter_config['value']
                filtered_data = filtered_data[filtered_data[column] == value]
        
        return filtered_data
    
    def fallback_analysis(self, query):
        """Fallback analysis when Gemini is not available"""
        query_lower = query.lower()
        
        # Provide basic insights based on common questions
        if 'total' in query_lower and 'customer' in query_lower:
            return {
                'type': 'analysis',
                'response': f"Total customers: {len(self.data)}. Breakdown by experience: {self.data['Customer_Type'].value_counts().to_dict()}",
                'data_summary': self.get_data_summary()
            }
        elif 'course' in query_lower and 'popular' in query_lower:
            top_courses = self.get_top_courses(5)
            return {
                'type': 'analysis', 
                'response': f"Most popular courses: {list(top_courses.keys())[:5]}",
                'data_summary': self.get_data_summary()
            }
        else:
            # Try basic filtering
            parsed = self.fallback_parse_query(query)
            filtered_data = self.fallback_execute_query(query)[0]
            return {
                'type': 'filter',
                'data': filtered_data,
                'response': f"Found {len(filtered_data)} customers matching your query",
                'interpretation': str(parsed)
            }
    
    def fallback_parse_query(self, query):
        """Fallback parsing method when Gemini is not available"""
        query_lower = query.lower()
        
        # Course-related queries (match specific phrases before generic ones)
        if 'try scuba' in query_lower or 'try dive' in query_lower or 'discover scuba' in query_lower:
            course_filter = 'try scuba'
        elif 'scuba' in query_lower:
            course_filter = 'scuba'
        elif 'snorkel' in query_lower:
            course_filter = 'snorkel'
        elif 'owd' in query_lower or 'open water' in query_lower:
            # Use a phrase that likely appears in the courses text
            course_filter = 'open water'
        elif 'advanced' in query_lower and 'diver' in query_lower:
            course_filter = 'advanced'
        elif 'free diving' in query_lower:
            course_filter = 'free diving'
        else:
            course_filter = None
            
        # Experience level queries
        if 'experienced' in query_lower or ('advanced' in query_lower and 'customer' in query_lower):
            experience_filter = 'Advanced'
        elif 'beginner' in query_lower or 'new' in query_lower:
            experience_filter = 'Beginner'
        else:
            experience_filter = None
            
        # Dive count queries
        dive_count = None
        if 'more than' in query_lower or '>' in query_lower:
            numbers = re.findall(r'\d+', query)
            if numbers:
                dive_count = ('>', int(numbers[0]))
        elif 'less than' in query_lower or '<' in query_lower:
            numbers = re.findall(r'\d+', query)
            if numbers:
                dive_count = ('<', int(numbers[0]))
                
        return {
            'course_filter': course_filter,
            'experience_filter': experience_filter,
            'dive_count': dive_count
        }
    
    def fallback_execute_query(self, query):
        """Fallback execution method when Gemini is not available"""
        parsed = self.fallback_parse_query(query)
        filtered_data = self.data.copy()
        
        # Apply course filter
        if parsed['course_filter']:
            course_filter = parsed['course_filter']
            filtered_data = filtered_data[
                filtered_data['Courses'].str.lower().str.contains(course_filter, na=False)
            ]
            
        # Apply experience filter
        if parsed['experience_filter']:
            filtered_data = filtered_data[filtered_data['Customer_Type'] == parsed['experience_filter']]
                
        # Apply dive count filter
        if parsed['dive_count']:
            operator, value = parsed['dive_count']
            if operator == '>':
                filtered_data = filtered_data[filtered_data['Dives'] > value]
            elif operator == '<':
                filtered_data = filtered_data[filtered_data['Dives'] < value]
                
        return filtered_data, parsed

# Market Basket Analysis Class
class MarketBasketAnalyzer:
    def __init__(self, data):
        self.data = data
        self.transactions = self.prepare_transactions()
        
    def prepare_transactions(self):
        """Convert course data into transaction format"""
        transactions = []
        for idx, row in self.data.iterrows():
            if pd.notna(row['Courses']):
                courses = [course.strip() for course in row['Courses'].split(',')]
                # Remove empty strings and standardize
                courses = [course for course in courses if course]
                if len(courses) > 1:  # Only include if multiple courses
                    transactions.append(courses)
        return transactions
    
    def calculate_support(self, itemset, transactions=None):
        """Calculate support for an itemset"""
        if transactions is None:
            transactions = self.transactions
            
        count = 0
        for transaction in transactions:
            if all(item in transaction for item in itemset):
                count += 1
        
        return count / len(transactions) if transactions else 0
    
    def generate_frequent_itemsets(self, min_support=0.1, max_length=3):
        """Generate frequent itemsets using Apriori algorithm"""
        # Get all unique items
        all_items = set()
        for transaction in self.transactions:
            all_items.update(transaction)
        
        all_items = list(all_items)
        frequent_itemsets = []
        
        # Generate 1-itemsets
        for item in all_items:
            support = self.calculate_support([item])
            if support >= min_support:
                frequent_itemsets.append({
                    'itemset': [item],
                    'support': support,
                    'count': int(support * len(self.transactions))
                })
        
        # Generate k-itemsets (k > 1)
        for k in range(2, max_length + 1):
            k_itemsets = []
            for combo in combinations(all_items, k):
                support = self.calculate_support(list(combo))
                if support >= min_support:
                    k_itemsets.append({
                        'itemset': list(combo),
                        'support': support,
                        'count': int(support * len(self.transactions))
                    })
            
            frequent_itemsets.extend(k_itemsets)
            
            # If no frequent k-itemsets, stop
            if not k_itemsets:
                break
        
        return frequent_itemsets
    
    def generate_association_rules(self, frequent_itemsets, min_confidence=0.5):
        """Generate association rules from frequent itemsets"""
        rules = []
        
        for itemset_data in frequent_itemsets:
            itemset = itemset_data['itemset']
            if len(itemset) < 2:
                continue
                
            # Generate all possible rules
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    consequent = [item for item in itemset if item not in antecedent]
                    
                    # Calculate confidence
                    antecedent_support = self.calculate_support(list(antecedent))
                    if antecedent_support == 0:
                        continue
                        
                    confidence = itemset_data['support'] / antecedent_support
                    
                    if confidence >= min_confidence:
                        # Calculate lift
                        consequent_support = self.calculate_support(consequent)
                        lift = confidence / consequent_support if consequent_support > 0 else 0
                        
                        rules.append({
                            'antecedent': list(antecedent),
                            'consequent': consequent,
                            'support': itemset_data['support'],
                            'confidence': confidence,
                            'lift': lift,
                            'count': itemset_data['count']
                        })
        
        # Sort by confidence and lift
        rules.sort(key=lambda x: (x['confidence'], x['lift']), reverse=True)
        return rules
    
    def get_recommendations_for_courses(self, completed_courses, rules, top_n=5):
        """Get course recommendations based on completed courses"""
        recommendations = []
        
        for rule in rules:
            # Check if customer has completed all antecedent courses
            if all(course in completed_courses for course in rule['antecedent']):
                # Check if consequent courses are not already completed
                new_courses = [course for course in rule['consequent'] if course not in completed_courses]
                
                if new_courses:
                    recommendations.append({
                        'recommended_courses': new_courses,
                        'based_on': rule['antecedent'],
                        'confidence': rule['confidence'],
                        'lift': rule['lift'],
                        'reasoning': f"Customers who took {', '.join(rule['antecedent'])} also took {', '.join(new_courses)} ({rule['confidence']:.1%} of the time)"
                    })
        
        # Remove duplicates and sort by confidence
        seen_courses = set()
        unique_recommendations = []
        
        for rec in recommendations:
            course_key = tuple(sorted(rec['recommended_courses']))
            if course_key not in seen_courses:
                seen_courses.add(course_key)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:top_n]

# Load data and initialize components
df = load_sample_data()

# Initialize AI Assistant with optional Gemini API key
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = None

# In sidebar, allow user to input API key
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Configuration")

api_key_input = st.sidebar.text_input(
    "Gemini API Key (Optional)", 
    type="password",
    value=st.session_state.gemini_api_key or "",
    help="Enter your Google Gemini API key for enhanced AI capabilities"
)

if api_key_input:
    st.session_state.gemini_api_key = api_key_input

# Initialize AI assistant and Market Basket Analyzer
ai_assistant = SmartDivingCRMAI(df, st.session_state.gemini_api_key)
market_analyzer = MarketBasketAnalyzer(df)

# Sidebar navigation
st.sidebar.title("🤿 Diving CRM")
page = st.sidebar.selectbox(
    "Navigate to:",
    ["Dashboard", "Customer Database", "Smart AI Assistant", "Market Basket Analysis", "Analytics"]
)

# Dashboard Page
if page == "Dashboard":
    st.markdown('<h1 class="main-header">🤿 Diving Center CRM Dashboard</h1>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f'<div class="metric-card"><h3>{len(df)}</h3><p>Total Customers</p></div>',
            unsafe_allow_html=True
        )
    
    with col2:
        total_dives = df['Dives'].sum()
        st.markdown(
            f'<div class="metric-card"><h3>{total_dives:,}</h3><p>Total Dives</p></div>',
            unsafe_allow_html=True
        )
    
    with col3:
        active_customers = len(df[df['Dives'] > 0])
        st.markdown(
            f'<div class="metric-card"><h3>{active_customers}</h3><p>Active Divers</p></div>',
            unsafe_allow_html=True
        )
    
    with col4:
        avg_dives = df[df['Dives'] > 0]['Dives'].mean()
        st.markdown(
            f'<div class="metric-card"><h3>{avg_dives:.1f}</h3><p>Avg Dives per Customer</p></div>',
            unsafe_allow_html=True
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Distribution by Experience")
        customer_dist = df['Customer_Type'].value_counts()
        fig_pie = px.pie(values=customer_dist.values, names=customer_dist.index, 
                        title="Customer Experience Levels")
        fig_pie.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Dive Count Distribution")
        fig_hist = px.histogram(df[df['Dives'] > 0], x='Dives', nbins=20, 
                               title="Distribution of Dive Counts")
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Course Analysis
    st.subheader("Popular Courses")
    all_courses = []
    for courses in df['Courses'].dropna():
        course_list = [course.strip() for course in courses.split(',')]
        all_courses.extend(course_list)
    
    course_counts = pd.Series(all_courses).value_counts().head(10)
    fig_bar = px.bar(x=course_counts.index, y=course_counts.values,
                     title="Top 10 Most Popular Courses")
    fig_bar.update_xaxes(tickangle=45)
    st.plotly_chart(fig_bar, use_container_width=True)

elif page == "Customer Database":
    st.header("📋 Customer Database")
    
    # Search and filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("Search customers:", placeholder="Name, email, or course...")
    
    with col2:
        experience_filter = st.selectbox("Filter by Experience:", 
                                       ['All', 'Beginner', 'Intermediate', 'Advanced'])
    
    with col3:
        min_dives = st.number_input("Minimum dives:", min_value=0, value=0)
    
    # Apply filters
    filtered_df = df.copy()
    
    if search_term:
        mask = (
            filtered_df['Full_Name'].str.contains(search_term, case=False, na=False) |
            filtered_df['Email'].str.contains(search_term, case=False, na=False) |
            filtered_df['Courses'].str.contains(search_term, case=False, na=False)
        )
        filtered_df = filtered_df[mask]
    
    if experience_filter != 'All':
        filtered_df = filtered_df[filtered_df['Customer_Type'] == experience_filter]
    
    if min_dives > 0:
        filtered_df = filtered_df[filtered_df['Dives'] >= min_dives]
    
    st.write(f"Showing {len(filtered_df)} customers")
    
    # Customer cards
    for idx, customer in filtered_df.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**{customer['Full_Name']}**")
                st.write(f"📧 {customer['Email']}")
                if customer['Phone'] and str(customer['Phone']) not in ['0', '', 'nan']:
                    st.write(f"📞 {customer['Phone']}")
            
            with col2:
                st.write(f"🤿 **{customer['Dives']} dives**")
                st.write(f"📊 {customer['Customer_Type']}")
            
            with col3:
                st.write(f"🎓 **Courses:**")
                courses = customer['Courses'] if pd.notna(customer['Courses']) else 'None'
                st.write(courses[:50] + "..." if len(str(courses)) > 50 else courses)
            
            st.divider()

elif page == "Smart AI Assistant":
    st.header("🧠 Smart AI Assistant")
    
    # Show AI status
    if ai_assistant.use_gemini:
        st.success("🚀 **Enhanced AI Mode** - Full data analysis and insights available")
        st.markdown("""
        **Ask ANY question about your customer data!**
        """)
    else:
        st.info("🔧 **Basic AI Mode** - For enhanced capabilities, add your Gemini API key in the sidebar")
    
    # Query input
    user_query = st.text_input(
        "Ask any question about your diving center data:", 
        placeholder="e.g., What insights can you provide about customer retention?"
    )
    
    if user_query:
        with st.spinner("🧠 Analyzing your data..."):
            try:
                result = ai_assistant.execute_smart_query(user_query)
                
                if result:
                    # Display AI Response
                    st.markdown(f'<div class="ai-response">', unsafe_allow_html=True)
                    st.write("🧠 **AI Analysis:**")
                    st.write(result['response'])
                    st.markdown('</div>', unsafe_allow_html=True)
                    # Handle different result types
                    if result['type'] == 'filter':
                        # Display filtered data
                        filtered_data = result['data']
                        st.subheader(f"📋 Results ({len(filtered_data)} customers found)")
                        
                        if len(filtered_data) > 0:
                            # Summary stats
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Customers", len(filtered_data))
                            with col2:
                                st.metric("Total Dives", filtered_data['Dives'].sum())
                            with col3:
                                avg_dives = filtered_data[filtered_data['Dives'] > 0]['Dives'].mean()
                                st.metric("Avg Dives", f"{avg_dives:.1f}" if not pd.isna(avg_dives) else "N/A")
                            with col4:
                                total_courses = filtered_data['Total_Courses'].sum()
                                st.metric("Total Courses", total_courses)
                            
                            # Display data
                            display_cols = ['Full_Name', 'Email', 'Dives', 'Customer_Type', 'Courses']
                            st.dataframe(filtered_data[display_cols], use_container_width=True)
                            
                            # Download option
                            csv = filtered_data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"customer_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No customers found matching your criteria.")
                    
                    elif result['type'] == 'analysis':
                        # Display analytical insights
                        st.markdown(f'<div class="insight-box">', unsafe_allow_html=True)
                        st.write("📊 **Key Statistics:**")
                        
                        data_summary = result['data_summary']
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"• Total Customers: {data_summary['total_customers']:,}")
                            st.write(f"• Total Dives: {data_summary['total_dives']:,}")
                            st.write(f"• Average Dives: {data_summary['average_dives']:.1f}")
                            st.write(f"• Email Completion: {data_summary['email_completion']:.1f}%")
                        
                        with col2:
                            st.write("Customer Distribution:")
                            for ctype, count in data_summary['customer_types'].items():
                                st.write(f"• {ctype}: {count}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Top courses chart
                        if data_summary['top_courses']:
                            st.subheader("📈 Top Courses")
                            courses = list(data_summary['top_courses'].keys())[:10]
                            counts = list(data_summary['top_courses'].values())[:10]
                            
                            fig = px.bar(x=courses, y=counts, title="Most Popular Courses")
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    else:  # recommendations
                        st.markdown(f'<div class="insight-box">', unsafe_allow_html=True)
                        st.write("💡 **Business Recommendations:**")
                        st.write("Based on your data analysis, here are actionable insights for your diving center.")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                else:
                    st.error("❌ Sorry, I couldn't process your query. Please try rephrasing it.")
                    
            except Exception as e:
                st.error(f"🚨 Error processing query: {str(e)}")

elif page == "Market Basket Analysis":
    st.header("🛒 Market Basket Analysis - Course Recommendations")
    
    st.markdown("""
    **Market Basket Analysis** helps identify which courses are frequently taken together, 
    enabling you to make smart recommendations to customers based on their completed courses.
    """)
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        min_support = st.slider("Minimum Support (% of customers)", 0.05, 0.5, 0.1, 0.05)
        st.caption("Higher values = more common course combinations")
    
    with col2:
        min_confidence = st.slider("Minimum Confidence (%)", 0.3, 0.9, 0.6, 0.1)
        st.caption("Higher values = stronger associations")
    
    # Persist analysis results across reruns
    if 'mba_frequent_itemsets' not in st.session_state:
        st.session_state.mba_frequent_itemsets = None
    if 'mba_association_rules' not in st.session_state:
        st.session_state.mba_association_rules = None
    if 'mba_params' not in st.session_state:
        st.session_state.mba_params = {}
    if 'mba_ready' not in st.session_state:
        st.session_state.mba_ready = False

    if st.button("🔄 Analyze Course Patterns", type="primary"):
        with st.spinner("Analyzing course combinations..."):
            # Generate frequent itemsets and association rules
            frequent_itemsets = market_analyzer.generate_frequent_itemsets(
                min_support=min_support, max_length=3
            )
            association_rules = market_analyzer.generate_association_rules(
                frequent_itemsets, min_confidence=min_confidence
            )
        # Store results in session state so UI selections don't clear them
        st.session_state.mba_frequent_itemsets = frequent_itemsets
        st.session_state.mba_association_rules = association_rules
        st.session_state.mba_params = {"min_support": min_support, "min_confidence": min_confidence}
        st.session_state.mba_ready = True
        if not association_rules:
            st.warning("No association rules found with current parameters. Try lowering the minimum support or confidence thresholds.")

    # Render previously computed analysis if available
    if st.session_state.mba_ready and st.session_state.mba_association_rules:
        association_rules = st.session_state.mba_association_rules
        frequent_itemsets = st.session_state.mba_frequent_itemsets

        st.success(f"Found {len(association_rules)} association rules!")
        
        # Display top association rules
        st.subheader("🎯 Top Association Rules")
        
        for i, rule in enumerate(association_rules[:10]):
            with st.container():
                st.markdown(f'<div class="rule-card">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    antecedent_str = " + ".join(rule['antecedent'])
                    consequent_str = " + ".join(rule['consequent'])
                    st.markdown(f"**If customer takes:** {antecedent_str}")
                    st.markdown(f"**Then they will likely take:** {consequent_str}")
                
                with col2:
                    st.metric("Confidence", f"{rule['confidence']:.1%}")
                    st.caption(f"Support: {rule['support']:.1%}")
                
                with col3:
                    st.metric("Lift", f"{rule['lift']:.2f}")
                    st.caption(f"Count: {rule['count']}")
                
                st.markdown('</div>', unsafe_allow_html=True)

        # Customer-specific recommendations
        st.subheader("🎓 Get Recommendations for Specific Customer")
        
        # Customer selector
        customer_options = df[df['Courses'].notna()]['Full_Name'].tolist()
        selected_customer = st.selectbox("Select a customer:", customer_options)
        
        if selected_customer:
            customer_data = df[df['Full_Name'] == selected_customer].iloc[0]
            completed_courses = [course.strip() for course in customer_data['Courses'].split(',')]
            
            st.write(f"**{selected_customer}** has completed:")
            st.write(", ".join(completed_courses))
            
            # Get recommendations
            recommendations = market_analyzer.get_recommendations_for_courses(
                completed_courses, association_rules, top_n=5
            )
            
            if recommendations:
                st.subheader("💡 Recommended Courses")
                
                for rec in recommendations:
                    with st.container():
                        st.markdown(f'<div class="rule-card">', unsafe_allow_html=True)
                        
                        recommended = ", ".join(rec['recommended_courses'])
                        based_on = ", ".join(rec['based_on'])
                        
                        st.markdown(f"**Recommend:** {recommended}")
                        st.markdown(f"**Based on:** {based_on}")
                        st.markdown(f"**Confidence:** {rec['confidence']:.1%} | **Lift:** {rec['lift']:.2f}")
                        st.caption(rec['reasoning'])
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No specific recommendations found for this customer's course combination.")

        # Frequent Course Combinations
        st.subheader("📊 Most Frequent Course Combinations")
        
        # Filter for combinations (length > 1)
        combinations = [item for item in frequent_itemsets if len(item['itemset']) > 1]
        combinations.sort(key=lambda x: x['support'], reverse=True)
        
        if combinations:
            combo_data = []
            for combo in combinations[:15]:
                combo_data.append({
                    'Course Combination': ' + '.join(combo['itemset']),
                    'Support (%)': f"{combo['support']:.1%}",
                    'Customer Count': combo['count']
                })
            
            combo_df = pd.DataFrame(combo_data)
            st.dataframe(combo_df, use_container_width=True)
            
            # Visualization
            fig = px.bar(
                combo_df.head(10), 
                x='Customer Count', 
                y='Course Combination',
                title="Top 10 Course Combinations by Popularity",
                orientation='h'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Business insights
        st.subheader("💼 Business Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Key Findings:**")
            if association_rules:
                strongest_rule = max(association_rules, key=lambda x: x['confidence'])
                st.write(f"• Strongest association: {' + '.join(strongest_rule['antecedent'])} → {' + '.join(strongest_rule['consequent'])} ({strongest_rule['confidence']:.1%})")
                
                highest_lift = max(association_rules, key=lambda x: x['lift'])
                st.write(f"• Most unexpected combo: {' + '.join(highest_lift['antecedent'])} → {' + '.join(highest_lift['consequent'])} (Lift: {highest_lift['lift']:.2f})")
                
                st.write(f"• Total course combinations analyzed: {len(market_analyzer.transactions)}")
        
        with col2:
            st.markdown("**Recommendations:**")
            st.write("• Use these insights for course package deals")
            st.write("• Target marketing based on completed courses") 
            st.write("• Train staff to suggest complementary courses")
            st.write("• Create course pathways for different skill levels")
    
    # Sample recommendations section
    with st.expander("📝 How to Use These Insights"):
        st.markdown("""
        **For Marketing:**
        - Create course bundles based on frequent combinations
        - Send targeted emails to customers who completed prerequisite courses
        - Offer discounts on complementary courses
        
        **For Operations:**
        - Schedule related courses closer together in time
        - Train instructors to mention related courses
        - Create learning pathways for different experience levels
        
        **For Customer Service:**
        - Proactively recommend next courses during completion ceremonies
        - Use recommendations in follow-up communications
        - Personalize the customer experience based on their course history
        """)

elif page == "Analytics":
    st.header("📊 Advanced Analytics")
    
    # Customer Segmentation
    st.subheader("Customer Segmentation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Dive experience vs course completion
        fig_scatter = px.scatter(df, x='Dives', y='Total_Courses', 
                                color='Customer_Type', size='Specialties',
                                title="Dives vs Courses Completed",
                                hover_data=['Full_Name'])
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Contact information completeness
        contact_data = {
            'Contact Type': ['Email', 'Phone', 'Mobile'],
            'Completeness': [
                len(df[df['Email'].notna()]) / len(df) * 100,
                len(df[df['Has_Phone'] == 1]) / len(df) * 100,
                len(df[df['Has_Mobile'] == 1]) / len(df) * 100
            ]
        }
        fig_contact = px.bar(contact_data, x='Contact Type', y='Completeness',
                            title="Contact Information Completeness (%)")
        st.plotly_chart(fig_contact, use_container_width=True)
    
    # Customer Value Analysis
    st.subheader("Customer Value Analysis")
    
    # Create customer value score based on dives and courses
    df['Value_Score'] = (df['Dives'] * 0.7) + (df['Total_Courses'] * 20) + (df['Specialties'] * 50)
    
    high_value = df[df['Value_Score'] > df['Value_Score'].quantile(0.8)]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_value = px.histogram(df, x='Value_Score', nbins=20,
                                title="Customer Value Score Distribution")
        st.plotly_chart(fig_value, use_container_width=True)
    
    with col2:
        st.subheader("High-Value Customers")
        st.write(f"Top 20% ({len(high_value)} customers)")
        
        for _, customer in high_value.head(5).iterrows():
            st.markdown(f"""
            **{customer['Full_Name']}**
            - 🤿 {customer['Dives']} dives
            - 🎓 {customer['Total_Courses']} courses
            - 📊 Score: {customer['Value_Score']:.1f}
            """)
    
    # Course progression analysis
    st.subheader("Course Progression Patterns")
    
    # Analyze common course sequences
    progression_data = []
    for idx, row in df.iterrows():
        if pd.notna(row['Courses']):
            courses = [course.strip() for course in row['Courses'].split(',')]
            if len(courses) >= 2:
                progression_data.append({
                    'Customer': row['Full_Name'],
                    'First_Course': courses[0],
                    'Last_Course': courses[-1],
                    'Total_Courses': len(courses),
                    'Dives': row['Dives']
                })
    
    if progression_data:
        prog_df = pd.DataFrame(progression_data)
        
        # Most common first courses
        first_courses = prog_df['First_Course'].value_counts().head(8)
        fig_first = px.bar(x=first_courses.index, y=first_courses.values,
                          title="Most Common First Courses")
        fig_first.update_xaxes(tickangle=45)
        st.plotly_chart(fig_first, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🤿 Enhanced Diving CRM System**")
st.sidebar.markdown("Built with Streamlit, AI & Market Basket Analysis")

# Instructions for deployment
st.sidebar.markdown("---")
with st.sidebar.expander("📖 Setup Instructions"):
    st.markdown("""
    **1. Install Required Packages:**
    ```bash
    pip install streamlit pandas plotly google-generativeai numpy
    ```
    
    **2. Get Gemini API Key:**
    - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
    - Create a free API key
    - Paste it in the sidebar above
    
    **3. Prepare Your CSV Data:**
    - Ensure your CSV has the required columns
    - Update the file path in the code
    
    **4. Run the App:**
    ```bash
    streamlit run enhanced_diving_crm.py
    ```
    
    **New Features:**
    - **Smart AI Assistant**: Ask any question about your data
    - **Market Basket Analysis**: Discover course patterns and recommendations
    - **Enhanced Analytics**: Better insights and visualizations
    """)
