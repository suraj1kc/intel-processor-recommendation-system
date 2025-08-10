import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Intel Processor Recommendation System",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #0066cc, #004499);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading and preprocessing
@st.cache_data
def load_and_process_data():
    """Load and preprocess the processor data"""
    try:
        df = pd.read_csv('data/intel_processors_features.csv')
        
        # Separate features
        feature_cols = [col for col in df.columns if col.startswith('feat.')]
        numerical_features = [col for col in feature_cols if col != 'feat.vertical_segment']
        
        # Create feature matrix for similarity calculation
        segment_encoded = pd.get_dummies(df['feat.vertical_segment'], prefix='segment')
        feature_matrix = pd.concat([df[numerical_features], segment_encoded], axis=1)
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(features_scaled)
        
        return df, feature_matrix, features_scaled, similarity_matrix, scaler
    except FileNotFoundError:
        st.error("❌ Could not find 'data/intel_processors_features.csv'. Please make sure the file is in the correct directory.")
        return None, None, None, None, None

def get_similar_processors(df, similarity_matrix, processor_name, top_n=5):
    """Find similar processors"""
    processor_names = df['processor_name'].tolist()
    
    if processor_name not in processor_names:
        return None
    
    processor_idx = processor_names.index(processor_name)
    similarities = similarity_matrix[processor_idx]
    
    # Get top similar processors (excluding self)
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    
    results = []
    for idx in similar_indices:
        proc_info = df.iloc[idx]
        results.append({
            'processor_name': proc_info['processor_name'],
            'similarity_score': similarities[idx],
            'category': proc_info['category'],
            'cores': proc_info['feat.total_cores'],
            'max_freq_ghz': proc_info['feat.max_turbo_ghz'],
            'price_usd': proc_info['feat.price_usd'],
            'use_case': proc_info['feat.vertical_segment']
        })
    
    return pd.DataFrame(results)

def filter_processors(df, **filters):
    """Filter processors based on criteria"""
    filtered_df = df.copy()
    
    for key, value in filters.items():
        if value is not None:
            if key == 'min_cores':
                filtered_df = filtered_df[filtered_df['feat.total_cores'] >= value]
            elif key == 'max_cores':
                filtered_df = filtered_df[filtered_df['feat.total_cores'] <= value]
            elif key == 'min_freq':
                filtered_df = filtered_df[filtered_df['feat.max_turbo_ghz'] >= value]
            elif key == 'max_freq':
                filtered_df = filtered_df[filtered_df['feat.max_turbo_ghz'] <= value]
            elif key == 'min_price':
                valid_prices = filtered_df['feat.price_usd'] != 1195.0
                filtered_df = filtered_df[valid_prices & (filtered_df['feat.price_usd'] >= value)]
            elif key == 'max_price':
                valid_prices = filtered_df['feat.price_usd'] != 1195.0
                filtered_df = filtered_df[valid_prices & (filtered_df['feat.price_usd'] <= value)]
            elif key == 'use_case' and value != 'All':
                filtered_df = filtered_df[filtered_df['feat.vertical_segment'] == value]
            elif key == 'category' and value != 'All':
                filtered_df = filtered_df[filtered_df['category'] == value]
    
    return filtered_df

def create_comparison_chart(processors_df):
    """Create a comparison chart for processors"""
    if len(processors_df) == 0:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance vs Price', 'Cores vs Frequency', 'Power Efficiency', 'Price Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Performance vs Price
    valid_prices = processors_df['feat.price_usd'] != 1195.0
    if valid_prices.any():
        price_data = processors_df[valid_prices]
        fig.add_trace(
            go.Scatter(
                x=price_data['feat.price_usd'],
                y=price_data['feat.max_turbo_ghz'],
                mode='markers',
                text=price_data['processor_name'].str[:30] + '...',
                name='Price vs Frequency',
                marker=dict(size=8, color='blue', opacity=0.7)
            ),
            row=1, col=1
        )
    
    # Cores vs Frequency
    fig.add_trace(
        go.Scatter(
            x=processors_df['feat.total_cores'],
            y=processors_df['feat.max_turbo_ghz'],
            mode='markers',
            text=processors_df['processor_name'].str[:30] + '...',
            name='Cores vs Frequency',
            marker=dict(size=8, color='red', opacity=0.7)
        ),
        row=1, col=2
    )
    
    # Power Efficiency
    fig.add_trace(
        go.Scatter(
            x=processors_df['feat.base_power_w'],
            y=processors_df['feat.freq_per_watt'],
            mode='markers',
            text=processors_df['processor_name'].str[:30] + '...',
            name='Power Efficiency',
            marker=dict(size=8, color='green', opacity=0.7)
        ),
        row=2, col=1
    )
    
    # Price Distribution
    if valid_prices.any():
        fig.add_trace(
            go.Histogram(
                x=price_data['feat.price_usd'],
                name='Price Distribution',
                nbinsx=20,
                marker=dict(color='orange', opacity=0.7)
            ),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=False, title_text="Processor Analysis Dashboard")
    fig.update_xaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Max Frequency (GHz)", row=1, col=1)
    fig.update_xaxes(title_text="Cores", row=1, col=2)
    fig.update_yaxes(title_text="Max Frequency (GHz)", row=1, col=2)
    fig.update_xaxes(title_text="Base Power (W)", row=2, col=1)
    fig.update_yaxes(title_text="Freq per Watt", row=2, col=1)
    fig.update_xaxes(title_text="Price (USD)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    return fig

def main():
    # Load data
    df, feature_matrix, features_scaled, similarity_matrix, scaler = load_and_process_data()
    
    if df is None:
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">🖥️ Intel Processor Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header" style="text-align: center;">Find the perfect Intel processor using AI-powered recommendations</p>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("🎯 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Recommendation Type",
        ["🏠 Home", "🔍 Find Similar Processors", "⚙️ Filter by Requirements", "💰 Best Value Options", "📊 Data Explorer"]
    )
    
    # Dataset overview in sidebar
    st.sidebar.markdown("### 📊 Dataset Overview")
    st.sidebar.info(f"""
    **Total Processors:** {len(df)}
    
    **Categories:**
    - Xeon: {len(df[df['category'] == 'Xeon_Processors'])}
    - Core Ultra: {len(df[df['category'] == 'Core_Ultra_Processors'])}
    - Core: {len(df[df['category'] == 'Core_Processors'])}
    - Xeon Max: {len(df[df['category'] == 'Xeon_Max_Processors'])}
    
    **Use Cases:**
    - Server: {len(df[df['feat.vertical_segment'] == 'Server'])}
    - Mobile: {len(df[df['feat.vertical_segment'] == 'Mobile'])}
    - Desktop: {len(df[df['feat.vertical_segment'] == 'Desktop'])}
    - Embedded: {len(df[df['feat.vertical_segment'] == 'Embedded'])}
    """)
    
    if app_mode == "🏠 Home":
        # Home page
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Processors", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            avg_cores = df['feat.total_cores'].mean()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average Cores", f"{avg_cores:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            avg_freq = df['feat.max_turbo_ghz'].mean()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average Max Frequency", f"{avg_freq:.1f} GHz")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🚀 How to Use This System")
        st.markdown("""
        1. **🔍 Find Similar Processors**: Enter a processor name to find similar alternatives
        2. **⚙️ Filter by Requirements**: Specify your needs (cores, frequency, price, use case)
        3. **💰 Best Value Options**: Find processors that offer the best performance per dollar
        4. **📊 Data Explorer**: Visualize and explore the processor dataset
        """)
        
        # Quick stats visualization
        fig = px.bar(
            df['category'].value_counts().reset_index(),
            x='index', y='category',
            title="Processor Distribution by Category",
            labels={'index': 'Category', 'category': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif app_mode == "🔍 Find Similar Processors":
        st.markdown("### 🔍 Find Similar Processors")
        st.markdown("Enter a processor name or select from the dropdown to find similar alternatives.")
        
        # Processor selection
        search_method = st.radio("How would you like to find processors?", ["Search by name", "Select from dropdown"])
        
        if search_method == "Search by name":
            search_term = st.text_input("🔎 Enter processor name or part of name:", placeholder="e.g., Core Ultra 5, Xeon 6780")
            
            if search_term:
                # Find matching processors
                matches = df[df['processor_name'].str.contains(search_term, case=False, na=False)]
                
                if len(matches) == 0:
                    st.error(f"❌ No processors found matching '{search_term}'")
                elif len(matches) == 1:
                    selected_processor = matches.iloc[0]['processor_name']
                else:
                    st.success(f"🎯 Found {len(matches)} processors matching '{search_term}'")
                    selected_processor = st.selectbox("Select a processor:", matches['processor_name'].tolist())
        else:
            selected_processor = st.selectbox("Select a processor:", df['processor_name'].tolist())
        
        if 'selected_processor' in locals() and selected_processor:
            # Show selected processor details
            proc_details = df[df['processor_name'] == selected_processor].iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📋 Selected Processor Details")
                st.info(f"""
                **Name:** {proc_details['processor_name'][:50]}...
                
                **Category:** {proc_details['category']}
                **Use Case:** {proc_details['feat.vertical_segment']}
                **Cores:** {proc_details['feat.total_cores']:.0f}
                **Max Frequency:** {proc_details['feat.max_turbo_ghz']:.1f} GHz
                **Price:** ${proc_details['feat.price_usd']:.0f}
                """)
            
            with col2:
                num_recommendations = st.slider("Number of recommendations:", 3, 10, 5)
            
            # Get recommendations
            recommendations = get_similar_processors(df, similarity_matrix, selected_processor, num_recommendations)
            
            if recommendations is not None:
                st.markdown("#### 🏆 Recommended Similar Processors")
                
                for i, row in recommendations.iterrows():
                    with st.expander(f"{i+1}. {row['processor_name'][:60]}... (Similarity: {row['similarity_score']:.3f})"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Cores", f"{row['cores']:.0f}")
                            st.metric("Category", row['category'])
                        with col2:
                            st.metric("Max Frequency", f"{row['max_freq_ghz']:.1f} GHz")
                            st.metric("Use Case", row['use_case'])
                        with col3:
                            st.metric("Price", f"${row['price_usd']:.0f}")
                            st.metric("Similarity Score", f"{row['similarity_score']:.3f}")
    
    elif app_mode == "⚙️ Filter by Requirements":
        st.markdown("### ⚙️ Filter Processors by Requirements")
        st.markdown("Specify your requirements to find processors that match your needs.")
        
        # Filter controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔧 Performance Requirements")
            min_cores = st.number_input("Minimum cores:", min_value=1, max_value=200, value=None, step=1)
            max_cores = st.number_input("Maximum cores:", min_value=1, max_value=200, value=None, step=1)
            min_freq = st.number_input("Minimum frequency (GHz):", min_value=1.0, max_value=8.0, value=None, step=0.1, format="%.1f")
            max_freq = st.number_input("Maximum frequency (GHz):", min_value=1.0, max_value=8.0, value=None, step=0.1, format="%.1f")
        
        with col2:
            st.markdown("#### 💰 Budget & Use Case")
            min_price = st.number_input("Minimum price (USD):", min_value=0, max_value=50000, value=None, step=50)
            max_price = st.number_input("Maximum price (USD):", min_value=0, max_value=50000, value=None, step=50)
            use_case = st.selectbox("Use case:", ['All'] + list(df['feat.vertical_segment'].unique()))
            category = st.selectbox("Category:", ['All'] + list(df['category'].unique()))
        
        # Apply filters
        filters = {
            'min_cores': min_cores,
            'max_cores': max_cores,
            'min_freq': min_freq,
            'max_freq': max_freq,
            'min_price': min_price,
            'max_price': max_price,
            'use_case': use_case,
            'category': category
        }
        
        filtered_df = filter_processors(df, **filters)
        
        if len(filtered_df) > 0:
            st.success(f"🎯 Found {len(filtered_df)} processors matching your requirements")
            
            # Sort options
            sort_by = st.selectbox("Sort by:", ['Price', 'Cores', 'Frequency', 'Performance per Dollar'])
            
            if sort_by == 'Performance per Dollar':
                valid_prices = filtered_df['feat.price_usd'] != 1195.0
                if valid_prices.any():
                    filtered_df = filtered_df[valid_prices].copy()
                    filtered_df['perf_per_dollar'] = (filtered_df['feat.max_turbo_ghz'] * filtered_df['feat.total_cores']) / filtered_df['feat.price_usd']
                    filtered_df = filtered_df.sort_values('perf_per_dollar', ascending=False)
            elif sort_by == 'Price':
                valid_prices = filtered_df['feat.price_usd'] != 1195.0
                filtered_df = filtered_df[valid_prices].sort_values('feat.price_usd')
            elif sort_by == 'Cores':
                filtered_df = filtered_df.sort_values('feat.total_cores', ascending=False)
            elif sort_by == 'Frequency':
                filtered_df = filtered_df.sort_values('feat.max_turbo_ghz', ascending=False)
            
            # Display results
            for i, (_, row) in enumerate(filtered_df.head(10).iterrows()):
                with st.expander(f"{i+1}. {row['processor_name'][:60]}..."):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Cores", f"{row['feat.total_cores']:.0f}")
                        st.metric("Threads", f"{row['feat.total_threads']:.0f}")
                    with col2:
                        st.metric("Max Frequency", f"{row['feat.max_turbo_ghz']:.1f} GHz")
                        st.metric("Base Frequency", f"{row['feat.base_freq_ghz']:.1f} GHz")
                    with col3:
                        st.metric("Price", f"${row['feat.price_usd']:.0f}")
                        st.metric("Use Case", row['feat.vertical_segment'])
                    with col4:
                        st.metric("Base Power", f"{row['feat.base_power_w']:.0f}W")
                        if 'perf_per_dollar' in row:
                            st.metric("Perf/Dollar", f"{row['perf_per_dollar']:.4f}")
            
            # Visualization
            if len(filtered_df) > 1:
                fig = create_comparison_chart(filtered_df.head(20))
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ No processors match your criteria. Try relaxing some constraints.")
    
    elif app_mode == "💰 Best Value Options":
        st.markdown("### 💰 Best Value Processors")
        st.markdown("Find processors that offer the best performance per dollar.")
        
        col1, col2 = st.columns(2)
        with col1:
            max_budget = st.number_input("Maximum budget (USD):", min_value=100, max_value=50000, value=1000, step=100)
        with col2:
            value_use_case = st.selectbox("Use case:", ['All'] + list(df['feat.vertical_segment'].unique()), key="value_use_case")
        
        # Filter by budget and use case
        filtered_df = filter_processors(df, max_price=max_budget, use_case=value_use_case)
        
        # Calculate value scores
        valid_prices = filtered_df['feat.price_usd'] != 1195.0
        value_df = filtered_df[valid_prices].copy()
        
        if len(value_df) > 0:
            value_df['value_score'] = (value_df['feat.max_turbo_ghz'] * value_df['feat.total_cores']) / value_df['feat.price_usd']
            value_df = value_df.sort_values('value_score', ascending=False)
            
            st.success(f"🎯 Found {len(value_df)} processors under ${max_budget}")
            
            # Display top value processors
            st.markdown("#### 🏆 Best Value Processors")
            for i, (_, row) in enumerate(value_df.head(8).iterrows()):
                with st.expander(f"{i+1}. {row['processor_name'][:60]}... (Value Score: {row['value_score']:.4f})"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Cores", f"{row['feat.total_cores']:.0f}")
                        st.metric("Frequency", f"{row['feat.max_turbo_ghz']:.1f} GHz")
                    with col2:
                        st.metric("Price", f"${row['feat.price_usd']:.0f}")
                        st.metric("Use Case", row['feat.vertical_segment'])
                    with col3:
                        st.metric("Value Score", f"{row['value_score']:.4f}")
                        st.metric("Power", f"{row['feat.base_power_w']:.0f}W")
                    with col4:
                        st.metric("Category", row['category'])
                        st.metric("Cache", f"{row['feat.cache_mb']:.0f}MB")
            
            # Value visualization
            fig = px.scatter(
                value_df.head(20),
                x='feat.price_usd',
                y='value_score',
                size='feat.total_cores',
                color='feat.vertical_segment',
                hover_data=['processor_name', 'feat.max_turbo_ghz'],
                title="Value Score vs Price",
                labels={'feat.price_usd': 'Price (USD)', 'value_score': 'Value Score (Performance/Dollar)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ No processors found in your budget range with valid pricing.")
    
    elif app_mode == "📊 Data Explorer":
        st.markdown("### 📊 Data Explorer")
        st.markdown("Explore and visualize the processor dataset.")
        
        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Processors", len(df))
        with col2:
            st.metric("Avg Cores", f"{df['feat.total_cores'].mean():.1f}")
        with col3:
            st.metric("Avg Frequency", f"{df['feat.max_turbo_ghz'].mean():.1f} GHz")
        with col4:
            valid_prices = df['feat.price_usd'] != 1195.0
            if valid_prices.any():
                st.metric("Avg Price", f"${df[valid_prices]['feat.price_usd'].mean():.0f}")
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "📋 Raw Data"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.histogram(df, x='feat.total_cores', title="Core Count Distribution")
                st.plotly_chart(fig1, use_container_width=True)
                
                fig3 = px.box(df, y='feat.max_turbo_ghz', x='feat.vertical_segment', title="Frequency by Use Case")
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                fig2 = px.histogram(df, x='feat.max_turbo_ghz', title="Frequency Distribution")
                st.plotly_chart(fig2, use_container_width=True)
                
                fig4 = px.scatter(df, x='feat.total_cores', y='feat.max_turbo_ghz', 
                                color='category', title="Cores vs Frequency by Category")
                st.plotly_chart(fig4, use_container_width=True)
        
        with tab2:
            # Correlation heatmap
            numeric_cols = [col for col in df.columns if col.startswith('feat.') and col != 'feat.vertical_segment']
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                          title="Feature Correlation Matrix",
                          color_continuous_scale='RdBu_r',
                          aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("#### 📋 Raw Dataset")
            search_col = st.selectbox("Search in column:", ['processor_name'] + list(df.columns))
            search_value = st.text_input(f"Search for value in {search_col}:")
            
            if search_value:
                filtered_data = df[df[search_col].astype(str).str.contains(search_value, case=False, na=False)]
                st.dataframe(filtered_data, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
