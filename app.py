import streamlit as st
from data_processor import DataProcessor
import config
import asyncio
from pathlib import Path
import logging
import json
import plotly.express as px
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styles
st.markdown("""
<style>
    /* Modern theme variables */
    :root {
        --primary: #4F46E5;
        --primary-dark: #4338CA;
        --secondary: #7C3AED;
        --bg-dark: #111827;
        --bg-card: #1F2937;
        --text-light: #F9FAFB;
        --text-muted: #E5E7EB;
        --radius-sm: 0.5rem;
        --radius-md: 0.75rem;
        --radius-lg: 1rem;
        --spacing-sm: 0.5rem;
        --spacing-md: 1rem;
        --spacing-lg: 1.5rem;
        --spacing-xl: 2rem;
        --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 8px 16px rgba(0, 0, 0, 0.15);
        --gradient-primary: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        --transition-all: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    @keyframes scaleIn {
        from { transform: scale(0.95); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }

    .animate-fade-in { animation: fadeIn 0.5s ease-out; }
    .animate-slide-in { animation: slideIn 0.5s ease-out; }
    .animate-scale-in { animation: scaleIn 0.3s ease-out; }

    /* Interactive button styles */
    .stButton > button {
        width: 100%;
        background: var(--gradient-primary);
        color: var(--text-light);
        border: none;
        padding: var(--spacing-md);
        border-radius: var(--radius-md);
        font-weight: 600;
        letter-spacing: 0.025em;
        transition: var(--transition-all);
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.2),
            transparent
        );
        transition: var(--transition-all);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }

    .stButton > button:hover::before {
        left: 100%;
        transition: 0.5s;
    }

    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: var(--shadow-sm);
    }

    /* Layout styles */
    .main .block-container {
        padding: var(--spacing-lg) var(--spacing-md);
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Enhanced card and container styles */
    .info-card {
        background: linear-gradient(180deg, var(--bg-card) 0%, rgba(31, 41, 55, 0.95) 100%);
        padding: var(--spacing-lg);
        border-radius: var(--radius-lg);
        margin-bottom: var(--spacing-lg);
        border: 1px solid rgba(79, 70, 229, 0.1);
        box-shadow: var(--shadow-md);
        transition: var(--transition-all);
        position: relative;
        overflow: hidden;
    }

    .info-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        width: 4px;
        height: 100%;
        background: var(--gradient-primary);
        border-radius: 4px 0 0 4px;
    }

    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }

    .metric-container {
        background: var(--bg-card);
        padding: var(--spacing-lg);
        border-radius: var(--radius-lg);
        margin-bottom: var(--spacing-lg);
        box-shadow: var(--shadow-md);
        transition: var(--transition-all);
        border: 1px solid rgba(79, 70, 229, 0.1);
        position: relative;
        overflow: hidden;
    }

    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }

    .metric-container::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: var(--gradient-primary);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.3s ease;
    }

    .metric-container:hover::after {
        transform: scaleX(1);
    }

    /* Section styles */
    .section-header {
        color: var(--primary-color);
        font-size: 1.5rem;
        font-weight: 600;
        margin: var(--spacing-lg) 0 var(--spacing-md);
        padding-bottom: var(--spacing-sm);
        border-bottom: 2px solid var(--primary-color);
    }

    /* File uploader styles */
    .file-uploader {
        border: 2px dashed var(--primary-color);
        border-radius: var(--border-radius);
        padding: var(--spacing-lg);
        text-align: center;
        background-color: var(--bg-card);
        transition: all 0.2s ease;
    }
    .file-uploader:hover {
        border-color: var(--primary-dark);
        background-color: rgba(31, 41, 55, 0.8);
    }

    /* Plot styles */
    .plot-container {
        background-color: var(--bg-card);
        padding: var(--spacing-md);
        border-radius: var(--border-radius);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Alert styles */
    .stAlert {
        padding: var(--spacing-md);
        margin-bottom: var(--spacing-md);
        border-radius: var(--border-radius);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Form styles */
    .stTextInput > div > div {
        border-radius: var(--border-radius);
    }
    
    .stSelectbox > div > div {
        border-radius: var(--border-radius);
    }

    /* Sidebar styles */
    .css-1d391kg {
        background-color: var(--bg-dark);
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'data_processor': None,
        'current_df': None,
        'last_uploaded_file': None,
        'processing_error': None,
        'processing_message': None,
        'api_key': None,
        'overview_data': None,
        'quick_insights': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Try to load API key from local storage
    if 'api_key' not in st.session_state or not st.session_state.api_key:
        try:
            with open('.deepseek_key', 'r') as f:
                st.session_state.api_key = f.read().strip()
        except:
            pass

def save_api_key(api_key: str):
    """Save API key to local storage"""
    try:
        with open('.deepseek_key', 'w') as f:
            f.write(api_key)
        st.session_state.api_key = api_key
        return True
    except Exception as e:
        logger.error(f"Failed to save API key: {str(e)}")
        return False

def setup_sidebar():
    """Setup sidebar navigation and controls"""
    with st.sidebar:
        st.title("🔍 " + config.APP_NAME)
        st.caption(f"Version {config.APP_VERSION}")
        
        # API Key Configuration
        with st.expander("⚙️ Configure API Key", expanded=not bool(st.session_state.api_key)):
            api_key = st.text_input(
                "DeepSeek API Key",
                type="password",
                value=st.session_state.api_key if st.session_state.api_key else "",
                help="Enter your DeepSeek API key to enable AI features"
            )
            if st.button("Save API Key", use_container_width=True):
                if save_api_key(api_key):
                    st.success("API key saved successfully")
                    st.rerun()
                else:
                    st.error("Failed to save API key")
        
        st.divider()
        
        # Feature list
        st.subheader("✨ Features")
        features = {
            "📊 Data Analysis": "Upload and analyze your data",
            "🤖 AI Insights": "Get AI-powered insights",
            "📈 Visualizations": "Create interactive plots",
            "📝 Data Stories": "Generate narrative reports"
        }
        for title, desc in features.items():
            st.markdown(f"**{title}**")
            st.caption(desc)

def process_uploaded_file(uploaded_file):
    """Process the uploaded file and update session state"""
    try:
        # Check if it's a new file
        if (st.session_state.last_uploaded_file != uploaded_file.name or
            st.session_state.current_df is None):
            
            with st.spinner("Processing data..."):
                df = st.session_state.data_processor.load_data(uploaded_file)
                if df is not None:
                    st.session_state.current_df = df
                    st.session_state.last_uploaded_file = uploaded_file.name
                    st.session_state.processing_message = "Data loaded successfully"
                    st.session_state.processing_error = None
                    
                    # Store overview data
                    st.session_state.overview_data = {
                        'total_records': len(df),
                        'columns': len(df.columns),
                        'quality': (1 - df.isnull().sum().sum()/(len(df)*len(df.columns)))*100
                    }
                    
                    # Generate and store quick insights
                    try:
                        quick_insights = asyncio.run(
                            st.session_state.data_processor.generate_insights(
                                df.head(1000)
                            )
                        )
                        st.session_state.quick_insights = quick_insights[:3]
                    except Exception as e:
                        logger.error(f"Error generating quick insights: {str(e)}")
                        st.session_state.quick_insights = []
                else:
                    st.session_state.processing_error = "Failed to load data"
                    return False
        
        return True
    except Exception as e:
        st.session_state.processing_error = f"Error processing file: {str(e)}"
        logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        return False

def render_file_uploader():
    """Render the file upload section"""
    uploaded_file = st.file_uploader(
        "📂 Upload your data file",
        type=list(config.ALLOWED_EXTENSIONS),
        key="data_file_uploader",
        help=f"Supported formats: {', '.join(config.ALLOWED_EXTENSIONS)}"
    )
    
    if uploaded_file:
        return process_uploaded_file(uploaded_file)
    return False

def render_sample_datasets():
    """Render sample datasets section"""
    with st.expander("📚 Sample Datasets", expanded=True):
        st.markdown("""
        <div class='card'>
            <h3 style='color: #4F46E5; margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.75rem;'>
                <span style='font-size: 1.5rem;'>📚</span>
                <span>Sample Datasets</span>
            </h3>
            <p style='color: var(--text-muted); margin-bottom: 1rem;'>Don't have a dataset? Try one of these:</p>
            <div style='display: flex; flex-direction: column; gap: 0.75rem;'>
                <a href='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' target='_blank' style='color: var(--text-light); text-decoration: none;'>
                    <div class='metric-card' style='text-align: left; padding: 1rem;'>
                        <div style='display: flex; align-items: center; gap: 0.75rem;'>
                            <span style='font-size: 1.25rem;'>🌸</span>
                            <span>Iris Dataset</span>
                        </div>
                    </div>
                </a>
                <a href='https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv' target='_blank' style='color: var(--text-light); text-decoration: none;'>
                    <div class='metric-card' style='text-align: left; padding: 1rem;'>
                        <div style='display: flex; align-items: center; gap: 0.75rem;'>
                            <span style='font-size: 1.25rem;'>🏠</span>
                            <span>Boston Housing</span>
                        </div>
                    </div>
                </a>
                <a href='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv' target='_blank' style='color: var(--text-light); text-decoration: none;'>
                    <div class='metric-card' style='text-align: left; padding: 1rem;'>
                        <div style='display: flex; align-items: center; gap: 0.75rem;'>
                            <span style='font-size: 1.25rem;'>🍷</span>
                            <span>Wine Quality</span>
                        </div>
                    </div>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_data_analysis():
    """Display data analysis section"""
    df = st.session_state.current_df
    
    # Display enhanced data overview
    overview_html = f"""
    <div class='card'>
        <h3 style='color: #4F46E5; margin: 0 0 2rem 0; display: flex; align-items: center; gap: 0.75rem;'>
            <span style='font-size: 1.5rem;'>📊</span>
            <span>Data Overview</span>
        </h3>
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem;'>
            <div class='metric-card'>
                <div class='metric-icon'>📝</div>
                <div class='metric-label'>Total Records</div>
                <div class='metric-value'>{len(df):,}</div>
                <div style='color: var(--text-muted); font-size: 0.875rem;'>Data points analyzed</div>
            </div>
            <div class='metric-card'>
                <div class='metric-icon'>🔢</div>
                <div class='metric-label'>Columns</div>
                <div class='metric-value'>{len(df.columns):,}</div>
                <div style='color: var(--text-muted); font-size: 0.875rem;'>Features available</div>
            </div>
            <div class='metric-card'>
                <div class='metric-icon'>💾</div>
                <div class='metric-label'>Memory Usage</div>
                <div class='metric-value'>{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}</div>
                <div style='color: var(--text-muted); font-size: 0.875rem;'>MB of data loaded</div>
            </div>
        </div>
    </div>
    """
    st.markdown(overview_html, unsafe_allow_html=True)
    
    # Enhanced data health score
    health_metrics = st.session_state.data_processor.calculate_health_score(df)
    if health_metrics:
        health_html = f"""
        <div class='card' style='margin-top: 2rem;'>
            <h3 style='color: #4F46E5; margin: 0 0 2rem 0; display: flex; align-items: center; gap: 0.75rem;'>
                <span style='font-size: 1.5rem;'>🏥</span>
                <span>Data Health Score</span>
            </h3>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem;'>
                <div class='metric-card'>
                    <div class='metric-icon'>💪</div>
                    <div class='metric-label'>Overall Health</div>
                    <div class='metric-value'>{health_metrics['health_score']:.1f}%</div>
                    <div style='color: var(--text-muted); font-size: 0.875rem;'>Data quality score</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-icon'>✅</div>
                    <div class='metric-label'>Completeness</div>
                    <div class='metric-value'>{health_metrics['completeness']:.1f}%</div>
                    <div style='color: var(--text-muted); font-size: 0.875rem;'>Data completeness rate</div>
                </div>
            </div>
        </div>
        """
        st.markdown(health_html, unsafe_allow_html=True)
        
        # Enhanced data preview
        preview_html = """
        <div class='card' style='margin-top: 2rem;'>
            <h3 style='color: #4F46E5; margin: 0 0 2rem 0; display: flex; align-items: center; gap: 0.75rem;'>
                <span style='font-size: 1.5rem;'>👀</span>
                <span>Data Preview</span>
            </h3>
            <div style='background: rgba(31, 41, 55, 0.5); padding: 1rem; border-radius: var(--radius-lg); border: 1px solid rgba(79, 70, 229, 0.1);'>
        """
        st.markdown(preview_html, unsafe_allow_html=True)
        st.dataframe(
            df.head(),
            use_container_width=True,
            height=300
        )
        st.markdown("</div></div>", unsafe_allow_html=True)

def display_visualization():
    """Display visualization section"""
    df = st.session_state.current_df
    
    st.markdown("""
    <div style='background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
        <h2 style='color: white; margin: 0;'>📈 Interactive Data Visualization</h2>
        <p style='color: #E5E7EB; margin: 0.5rem 0 0 0;'>Create custom visualizations to explore your data</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        # Chart configuration
        st.markdown("""
        <div style='background-color: #1F2937; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
            <h3 style='color: #4F46E5; margin: 0 0 1rem 0;'>🎨 Chart Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            chart_type = st.selectbox(
                "📊 Chart Type",
                ["scatter", "line", "bar", "histogram", "box"],
                help="Select the type of visualization"
            )
        with col2:
            x_col = st.selectbox(
                "📏 X-axis",
                df.columns,
                help="Select the column for X-axis"
            )
        with col3:
            y_col = st.selectbox(
                "📐 Y-axis",
                df.columns,
                help="Select the column for Y-axis"
            )
        
        # Advanced options in an expander
        with st.expander("🛠️ Advanced Options", expanded=False):
            color_col = st.selectbox(
                "🎨 Color by",
                ["None"] + list(df.columns),
                help="Select a column to color the data points"
            )
            color_col = None if color_col == "None" else color_col
        
        # Progress bar for visualization
        with st.spinner("Creating visualization..."):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            
            fig = st.session_state.data_processor.visualize_data(
                df, chart_type, x_col, y_col, color_col
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                
                # Download options
                st.download_button(
                    label="📥 Download Plot as HTML",
                    data=fig.to_html(),
                    file_name="visualization.html",
                    mime="text/html"
                )

def display_ai_features():
    """Display AI insights and data story sections"""
    df = st.session_state.current_df
    
    st.markdown("""
    <div style='background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
        <h2 style='color: white; margin: 0;'>🤖 AI-Powered Analysis</h2>
        <p style='color: #E5E7EB; margin: 0.5rem 0 0 0;'>Get deep insights and narratives about your data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for insights and story
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: #1F2937; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
            <h3 style='color: #4F46E5; margin: 0;'>💡 Key Insights</h3>
            <p style='color: #E5E7EB; margin: 0.5rem 0 0 0;'>Discover patterns and trends</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Generate Insights", use_container_width=True):
            with st.spinner("AI is analyzing your data..."):
                try:
                    # Add progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                    
                    # Generate comprehensive insights
                    insights = asyncio.run(
                        st.session_state.data_processor.generate_insights(df)
                    )
                    
                    # Group insights by category
                    categories = {
                        "Key Findings": [],
                        "Patterns & Trends": [],
                        "Recommendations": [],
                        "Forecasts": []
                    }
                    
                    for idx, insight in enumerate(insights):
                        # Categorize insights based on content
                        text = insight['insight'].lower()
                        if any(word in text for word in ['predict', 'forecast', 'future', 'expect']):
                            categories["Forecasts"].append(insight)
                        elif any(word in text for word in ['should', 'recommend', 'consider', 'suggest']):
                            categories["Recommendations"].append(insight)
                        elif any(word in text for word in ['pattern', 'trend', 'correlation', 'relationship']):
                            categories["Patterns & Trends"].append(insight)
                        else:
                            categories["Key Findings"].append(insight)
                    
                    # Display insights by category with icons
                    icons = {
                        "Key Findings": "💡",
                        "Patterns & Trends": "📈",
                        "Recommendations": "🎯",
                        "Forecasts": "🔮"
                    }
                    
                    for category, category_insights in categories.items():
                        if category_insights:
                            st.markdown(f"""
                            <div style='
                                background-color: #1F2937;
                                padding: 1rem;
                                border-radius: 0.5rem;
                                margin-bottom: 1rem;
                                border-left: 4px solid #4F46E5;
                                animation: fadeIn 0.5s ease-out;
                            '>
                                <h3 style='color: #4F46E5; margin: 0;'>{icons[category]} {category}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for idx, insight in enumerate(category_insights):
                                st.markdown(f"""
                                <div style='
                                    background-color: #1F2937;
                                    padding: 1rem;
                                    border-radius: 0.5rem;
                                    margin-bottom: 0.5rem;
                                    border-left: 4px solid #7C3AED;
                                    animation: fadeIn 0.5s ease-in-out {idx * 0.2}s;
                                '>
                                    <p style='color: #F9FAFB; margin: 0;'>{insight['insight']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Add visualizations for patterns and trends
                            if category == "Patterns & Trends":
                                try:
                                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                                    if len(numeric_cols) >= 2:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            # Correlation heatmap
                                            fig = px.imshow(
                                                df[numeric_cols].corr(),
                                                color_continuous_scale='RdBu_r',
                                                title="Correlation Heatmap"
                                            )
                                            fig.update_layout(
                                                plot_bgcolor='#1F2937',
                                                paper_bgcolor='#1F2937',
                                                font_color='#F9FAFB'
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                        
                                        with col2:
                                            # Distribution plot
                                            fig = px.box(
                                                df,
                                                y=numeric_cols[0],
                                                title=f"Distribution of {numeric_cols[0]}"
                                            )
                                            fig.update_layout(
                                                plot_bgcolor='#1F2937',
                                                paper_bgcolor='#1F2937',
                                                font_color='#F9FAFB'
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    logger.error(f"Error creating visualizations: {str(e)}")
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
                    logger.error(f"Error generating insights: {str(e)}")
    
    with col2:
        st.markdown("""
        <div style='background-color: #1F2937; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
            <h3 style='color: #4F46E5; margin: 0;'>📝 Data Story</h3>
            <p style='color: #E5E7EB; margin: 0.5rem 0 0 0;'>Get a narrative analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📖 Generate Story", use_container_width=True):
            with st.spinner("Crafting your data story..."):
                try:
                    # Add progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                    
                    story = asyncio.run(
                        st.session_state.data_processor.generate_data_story(df)
                    )
                    st.markdown(f"""
                    <div style='
                        background-color: #1F2937;
                        padding: 1.5rem;
                        border-radius: 0.5rem;
                        border-left: 4px solid #7C3AED;
                        animation: fadeIn 0.5s ease-in-out;
                    '>
                        <div style='color: #F9FAFB; line-height: 1.6;'>{story}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add download button for the story
                    st.download_button(
                        label="📥 Download Story",
                        data=story,
                        file_name="data_story.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Error generating data story: {str(e)}")
                    logger.error(f"Error generating data story: {str(e)}")

def display_sample_datasets():
    """Display sample datasets section"""
    st.markdown('<div class="section-header">📚 Sample Datasets</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
    Don't have a dataset? Try one of these:
    - 🌸 [Iris Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)
    - 🏠 [Boston Housing](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv)
    - 🍷 [Wine Quality](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    try:
        # Initialize application state
        init_session_state()
        
        # Setup sidebar
        setup_sidebar()
        
        # Check API key configuration
        if not st.session_state.api_key:
            st.warning("⚠️ Please configure your DeepSeek API key to use AI features")
        
        # Initialize data processor if needed
        if not st.session_state.data_processor:
            st.session_state.data_processor = DataProcessor(st.session_state.api_key)
        
        # Main header
        st.markdown("""
        <div class='header'>
            <h1>📊 Data Analysis Dashboard</h1>
            <p>Analyze your data with AI-powered insights</p>
        </div>
        """, unsafe_allow_html=True)
        # Create tabs with modern styling
        st.markdown("""
        <style>
        /* Enhanced tab styling */
        .stTabs {
            background: linear-gradient(180deg, #1F2937 0%, rgba(31, 41, 55, 0.95) 100%);
            padding: 1rem;
            border-radius: var(--radius-lg);
            margin-bottom: 2rem;
            box-shadow: var(--shadow-md);
            border: 1px solid rgba(79, 70, 229, 0.1);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.75rem;
            background-color: rgba(17, 24, 39, 0.5);
            padding: 0.5rem;
            border-radius: var(--radius-lg);
            border: 1px solid rgba(79, 70, 229, 0.1);
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1.25rem;
            border-radius: var(--radius-md);
            font-weight: 500;
            color: var(--text-muted);
            background-color: transparent;
            border: 1px solid transparent;
            transition: var(--transition-all);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(79, 70, 229, 0.1);
            color: var(--text-light);
            border-color: rgba(79, 70, 229, 0.2);
            transform: translateY(-1px);
        }

        .stTabs [aria-selected="true"] {
            background: var(--gradient-primary) !important;
            color: white !important;
            font-weight: 600;
            box-shadow: var(--shadow-md);
            border: none;
        }

        .stTabs [aria-selected="true"]:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-lg);
        }
        
        /* Enhanced card and metric components */
        .card {
            background: linear-gradient(180deg, var(--bg-card) 0%, rgba(31, 41, 55, 0.95) 100%);
            padding: var(--spacing-xl);
            border-radius: var(--radius-lg);
            margin-bottom: var(--spacing-lg);
            box-shadow: var(--shadow-md);
            transition: var(--transition-all);
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(79, 70, 229, 0.1);
        }

        .card::before {
            content: '';
            position: absolute;
            inset: 0;
            background: var(--gradient-primary);
            opacity: 0.05;
            transition: var(--transition-all);
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        .card:hover::before {
            opacity: 0.1;
        }

        /* Enhanced metric cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(31, 41, 55, 0.95) 0%, rgba(17, 24, 39, 0.98) 100%);
            padding: var(--spacing-xl);
            border-radius: var(--radius-lg);
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: var(--transition-all);
            animation: scaleIn 0.3s ease-out;
            border: 1px solid rgba(79, 70, 229, 0.1);
            box-shadow: var(--shadow-md);
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: var(--gradient-primary);
            opacity: 0;
            transition: var(--transition-all);
        }

        .metric-card::after {
            content: '';
            position: absolute;
            inset: 0;
            background: var(--gradient-primary);
            opacity: 0;
            transition: var(--transition-all);
            z-index: -1;
        }

        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
        }

        .metric-card:hover::before {
            opacity: 1;
        }

        .metric-card:hover::after {
            opacity: 0.05;
        }

        .metric-label {
            color: var(--text-muted);
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: var(--spacing-md);
            position: relative;
            display: inline-block;
        }

        .metric-label::after {
            content: '';
            position: absolute;
            bottom: -4px;
            left: 0;
            width: 100%;
            height: 2px;
            background: var(--gradient-primary);
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.3s ease;
        }

        .metric-card:hover .metric-label::after {
            transform: scaleX(1);
        }

        .metric-value {
            color: var(--text-light);
            font-size: 2.5rem;
            font-weight: 700;
            text-shadow: var(--shadow-sm);
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: var(--spacing-sm) 0;
            line-height: 1.2;
        }

        .metric-icon {
            font-size: 1.5rem;
            margin-bottom: var(--spacing-sm);
            color: var(--primary);
            background: rgba(79, 70, 229, 0.1);
            width: 3rem;
            height: 3rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto var(--spacing-md);
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Header styling */
        .header {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            padding: 2rem;
            border-radius: 0.75rem;
            margin-bottom: 1.5rem;
            color: white;
        }
        .header h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 600;
        }
        .header p {
            margin: 0.5rem 0 0 0;
            color: #E5E7EB;
            font-size: 1.1rem;
        }
        }
        </style>
        """, unsafe_allow_html=True)
        
        overview_tab, analysis_tab, visualization_tab, ai_tab = st.tabs([
            "📋 Overview",
            "📊 Analysis",
            "📈 Visualization",
            "🤖 AI Insights"
        ])
        
        with overview_tab:
            # Enhanced file upload styling
            st.markdown("""
            <style>
            /* File upload container */
            .stFileUploader {
                background: linear-gradient(135deg, rgba(31, 41, 55, 0.95) 0%, rgba(17, 24, 39, 0.98) 100%);
                padding: var(--spacing-lg);
                border-radius: var(--radius-lg);
                border: 1px solid rgba(79, 70, 229, 0.1);
                transition: var(--transition-all);
            }

            .stFileUploader:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-lg);
                border-color: rgba(79, 70, 229, 0.2);
            }

            /* Upload button */
            .stFileUploader > div > button {
                background: var(--gradient-primary) !important;
                color: white !important;
                border: none !important;
                padding: var(--spacing-md) var(--spacing-lg) !important;
                font-weight: 600 !important;
                letter-spacing: 0.025em !important;
                transition: var(--transition-all) !important;
            }

            .stFileUploader > div > button:hover {
                transform: translateY(-1px);
                box-shadow: var(--shadow-lg);
            }

            /* File info */
            .stFileUploader > div > small {
                color: var(--text-muted) !important;
                font-size: 0.875rem !important;
            }

            /* Success message */
            .stAlert {
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%) !important;
                border: 1px solid rgba(16, 185, 129, 0.2) !important;
                color: #10B981 !important;
            }

            /* Error message */
            .stAlert.error {
                background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%) !important;
                border: 1px solid rgba(239, 68, 68, 0.2) !important;
                color: #EF4444 !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Main content layout
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Upload section with modern styling
                st.markdown("""
                <div class='card'>
                    <h3 style='color: #4F46E5; margin: 0 0 1.5rem 0; display: flex; align-items: center; gap: 0.75rem;'>
                        <span style='font-size: 1.5rem;'>🚀</span>
                        <span>Get Started</span>
                    </h3>
                    <p style='color: var(--text-muted); margin-bottom: 1.5rem;'>Upload your data to unlock AI-powered insights</p>
                    <div style='background: rgba(31, 41, 55, 0.5); padding: 1.5rem; border-radius: var(--radius-lg); border: 1px solid rgba(79, 70, 229, 0.1);'>
                """, unsafe_allow_html=True)
                
                # File uploader
                if render_file_uploader():
                    if st.session_state.processing_message:
                        st.success(st.session_state.processing_message)
                        st.session_state.processing_message = None
                elif st.session_state.processing_error:
                    st.error(st.session_state.processing_error)
                    st.session_state.processing_error = None
                
                st.markdown("</div></div>", unsafe_allow_html=True)
                
                # Display data overview and insights
                if st.session_state.current_df is not None:
                    try:
                        df = st.session_state.current_df
                        
                        # Calculate metrics
                        total_records = len(df)
                        total_columns = len(df.columns)
                        quality_score = (1 - df.isnull().sum().sum()/(total_records*total_columns))*100
                        
                        # Display overview
                        overview_html = (
                            "<div class='card'>"
                            "<h3 style='color: #4F46E5; margin: 0 0 1.5rem 0;'>📊 Quick Overview</h3>"
                            "<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>"
                            f"<div class='metric-card'><div class='metric-label'>Total Records</div><div class='metric-value'>{total_records:,}</div></div>"
                            f"<div class='metric-card'><div class='metric-label'>Columns</div><div class='metric-value'>{total_columns:,}</div></div>"
                            f"<div class='metric-card'><div class='metric-label'>Data Quality</div><div class='metric-value'>{quality_score:.1f}%</div></div>"
                            "</div></div>"
                        )
                        st.markdown(overview_html, unsafe_allow_html=True)
                        
                        # Display insights
                        if st.session_state.quick_insights:
                            insights_html = (
                                "<div class='card' style='margin-top: 1.5rem;'>"
                                "<h3 style='color: #4F46E5; margin: 0 0 1rem 0;'>🤖 Quick AI Insights</h3>"
                                "<div style='display: flex; flex-direction: column; gap: 1rem;'>"
                            )
                            
                            for idx, insight in enumerate(st.session_state.quick_insights):
                                insights_html += (
                                    f"<div class='metric-card' style='text-align: left; animation-delay: {idx * 0.2}s;'>"
                                    f"<p style='color: #F9FAFB; margin: 0; line-height: 1.6;'>{insight['insight']}</p>"
                                    "</div>"
                                )
                            
                            insights_html += "</div></div>"
                            st.markdown(insights_html, unsafe_allow_html=True)
                    except Exception as e:
                        logger.error(f"Error displaying data overview: {str(e)}")
                        st.error("An error occurred while displaying the data overview")
                else:
                    if st.session_state.processing_error:
                        st.error(st.session_state.processing_error)
                        st.session_state.processing_error = None
                    else:
                        st.info("📂 Drag and drop your data file here")
                st.markdown('</div>', unsafe_allow_html=True)
                # Sample datasets section
                st.markdown("""
                <div class='card' style='margin-top: 1.5rem;'>
                    <h3 style='color: #4F46E5; margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.75rem;'>
                        <span style='font-size: 1.5rem;'>📚</span>
                        <span>Sample Datasets</span>
                    </h3>
                    <p style='color: var(--text-muted); margin-bottom: 1rem;'>Don't have a dataset? Try one of these:</p>
                    <div style='display: flex; flex-direction: column; gap: 0.75rem;'>
                        <a href='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' target='_blank' style='color: var(--text-light); text-decoration: none;'>
                            <div class='metric-card' style='text-align: left; padding: 1rem;'>
                                <div style='display: flex; align-items: center; gap: 0.75rem;'>
                                    <span style='font-size: 1.25rem;'>🌸</span>
                                    <span>Iris Dataset</span>
                                </div>
                            </div>
                        </a>
                        <a href='https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv' target='_blank' style='color: var(--text-light); text-decoration: none;'>
                            <div class='metric-card' style='text-align: left; padding: 1rem;'>
                                <div style='display: flex; align-items: center; gap: 0.75rem;'>
                                    <span style='font-size: 1.25rem;'>🏠</span>
                                    <span>Boston Housing</span>
                                </div>
                            </div>
                        </a>
                        <a href='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv' target='_blank' style='color: var(--text-light); text-decoration: none;'>
                            <div class='metric-card' style='text-align: left; padding: 1rem;'>
                                <div style='display: flex; align-items: center; gap: 0.75rem;'>
                                    <span style='font-size: 1.25rem;'>🍷</span>
                                    <span>Wine Quality</span>
                                </div>
                            </div>
                        </a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Features and Next Steps sections
                st.markdown("""
                <div class='card'>
                    <h3 style='color: #4F46E5; margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.75rem;'>
                        <span style='font-size: 1.5rem;'>✨</span>
                        <span>What You'll Get</span>
                    </h3>
                    <div style='display: flex; flex-direction: column; gap: 0.75rem;'>
                        <div class='metric-card' style='text-align: left; padding: 1rem;'>
                            <div style='display: flex; align-items: center; gap: 0.75rem;'>
                                <span style='font-size: 1.25rem;'>🤖</span>
                                <span>AI-powered data insights</span>
                            </div>
                        </div>
                        <div class='metric-card' style='text-align: left; padding: 1rem;'>
                            <div style='display: flex; align-items: center; gap: 0.75rem;'>
                                <span style='font-size: 1.25rem;'>📊</span>
                                <span>Interactive visualizations</span>
                            </div>
                        </div>
                        <div class='metric-card' style='text-align: left; padding: 1rem;'>
                            <div style='display: flex; align-items: center; gap: 0.75rem;'>
                                <span style='font-size: 1.25rem;'>📈</span>
                                <span>Data health analysis</span>
                            </div>
                        </div>
                        <div class='metric-card' style='text-align: left; padding: 1rem;'>
                            <div style='display: flex; align-items: center; gap: 0.75rem;'>
                                <span style='font-size: 1.25rem;'>📝</span>
                                <span>Narrative data stories</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class='card' style='margin-top: 1.5rem;'>
                    <h3 style='color: #4F46E5; margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.75rem;'>
                        <span style='font-size: 1.5rem;'>🎯</span>
                        <span>Next Steps</span>
                    </h3>
                    <p style='color: var(--text-muted); margin-bottom: 1rem;'>After uploading your data:</p>
                    <div style='display: flex; flex-direction: column; gap: 0.75rem;'>
                        <div class='metric-card' style='text-align: left; padding: 1rem;'>
                            <div style='display: flex; align-items: center; gap: 0.75rem;'>
                                <span style='font-size: 1.25rem;'>1️⃣</span>
                                <span>View data analysis in the Analysis tab</span>
                            </div>
                        </div>
                        <div class='metric-card' style='text-align: left; padding: 1rem;'>
                            <div style='display: flex; align-items: center; gap: 0.75rem;'>
                                <span style='font-size: 1.25rem;'>2️⃣</span>
                                <span>Create visualizations in the Visualization tab</span>
                            </div>
                        </div>
                        <div class='metric-card' style='text-align: left; padding: 1rem;'>
                            <div style='display: flex; align-items: center; gap: 0.75rem;'>
                                <span style='font-size: 1.25rem;'>3️⃣</span>
                                <span>Get AI insights in the AI Insights tab</span>
                            </div>
                        </div>
                    </div>
                </div>
                </div>
                
                <div style='
                    background-color: #1F2937;
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    border-left: 4px solid #7C3AED;
                '>
                    <h3 style='color: #7C3AED; margin: 0;'>🎯 Next Steps</h3>
                    <p style='color: #F9FAFB; margin: 0.5rem 0;'>After uploading:</p>
                    <ol style='color: #F9FAFB; margin: 0.5rem 0 0 0; padding-left: 1.5rem;'>
                        <li>View data analysis in the Analysis tab</li>
                        <li>Create visualizations in the Visualization tab</li>
                        <li>Get AI insights in the AI Insights tab</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
        
        # Add animation styles
        st.markdown("""
        <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        .animate {
            animation: fadeIn 0.5s ease-out forwards;
        }
        </style>
        """, unsafe_allow_html=True)

        # Handle tab content
        with analysis_tab:
            if st.session_state.current_df is not None:
                display_data_analysis()
            else:
                st.markdown("""
                <div style='
                    background-color: #1F2937;
                    padding: 2rem;
                    border-radius: 0.5rem;
                    text-align: center;
                    margin-top: 2rem;
                    animation: fadeIn 0.5s ease-out;
                '>
                    <h2 style='color: #4F46E5; margin: 0;'>📊 Data Analysis</h2>
                    <p style='color: #F9FAFB; margin: 1rem 0;'>Upload your data in the Overview tab to see:</p>
                    <div style='display: inline-block; text-align: left; color: #E5E7EB; margin-top: 1rem;'>
                        • Data health score<br>
                        • Column statistics<br>
                        • Data quality metrics<br>
                        • Distribution analysis
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with visualization_tab:
            if st.session_state.current_df is not None:
                display_visualization()
            else:
                st.markdown("""
                <div style='
                    background-color: #1F2937;
                    padding: 2rem;
                    border-radius: 0.5rem;
                    text-align: center;
                    margin-top: 2rem;
                    animation: fadeIn 0.5s ease-out;
                '>
                    <h2 style='color: #7C3AED; margin: 0;'>📈 Interactive Visualizations</h2>
                    <p style='color: #F9FAFB; margin: 1rem 0;'>Create custom visualizations after uploading:</p>
                    <div style='display: inline-block; text-align: left; color: #E5E7EB; margin-top: 1rem;'>
                        • Scatter plots<br>
                        • Line charts<br>
                        • Bar graphs<br>
                        • Histograms & Box plots
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with ai_tab:
            if st.session_state.current_df is not None:
                display_ai_features()
            else:
                st.markdown("""
                <div style='
                    background-color: #1F2937;
                    padding: 2rem;
                    border-radius: 0.5rem;
                    text-align: center;
                    margin-top: 2rem;
                    animation: fadeIn 0.5s ease-out;
                '>
                    <h2 style='color: #4F46E5; margin: 0;'>🤖 AI-Powered Insights</h2>
                    <p style='color: #F9FAFB; margin: 1rem 0;'>Get deep insights after uploading:</p>
                    <div style='display: inline-block; text-align: left; color: #E5E7EB; margin-top: 1rem;'>
                        • Pattern discovery<br>
                        • Trend analysis<br>
                        • Correlation insights<br>
                        • Narrative data stories
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Unexpected error in main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()