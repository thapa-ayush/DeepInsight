import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import streamlit as st
import asyncio
from pathlib import Path
import json
import aiohttp
import logging
from concurrent.futures import ThreadPoolExecutor
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, CACHE_TTL, logger

class DataProcessor:
    """Process and analyze data with optimized performance"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._setup_async()

    def _setup_async(self):
        """Setup async event loop"""
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def _validate_file(self, file) -> bool:
        """Validate file before processing"""
        try:
            file_extension = Path(file.name).suffix.lower()[1:]
            
            if not file_extension:
                st.error("File has no extension")
                return False
                
            if file_extension not in ALLOWED_EXTENSIONS:
                st.error(f"Unsupported file format: {file_extension}")
                return False
            
            if file.size > MAX_FILE_SIZE:
                st.error(f"File size exceeds maximum limit of {MAX_FILE_SIZE/1024/1024:.1f}MB")
                return False
                
            return True
        except Exception as e:
            st.error(f"Error validating file: {str(e)}")
            return False

    @st.cache_data(ttl=CACHE_TTL)
    def load_data(_self, file) -> pd.DataFrame:
        """Load and process data from various file formats"""
        if not _self._validate_file(file):
            return None
            
        try:
            file_extension = Path(file.name).suffix.lower()[1:]
            
            if file_extension in ['csv', 'txt', 'tsv']:
                df = pd.read_csv(file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file)
            elif file_extension == 'parquet':
                df = pd.read_parquet(file)
            elif file_extension == 'json':
                df = pd.read_json(file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
            
            if df.empty:
                st.error("The file contains no data")
                return None
                
            return _self._clean_data(df)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            logger.error(f"Error loading data from {file.name}: {str(e)}")
            return None

    def _parse_insights(self, response_text: str) -> List[Dict[str, str]]:
        """Parse and structure insights from API response with [Insight X.X] format"""
        insights = []
        current_category = None
        current_text = []
        
        # Define category mappings
        categories = {
            '1': 'Key Findings',
            '2': 'Business Impact',
            '3': 'Future Predictions',
            '4': 'Actionable Recommendations'
        }
        
        for line in response_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for category headers with more flexible matching
            if any(cat in line.lower() for cat in ['key findings:', 'business impact:', 'future predictions:', 'actionable recommendations:']):
                # Save previous insight if exists
                if current_text:
                    insights.append({
                        'category': current_category,
                        'insight': ' '.join(current_text).strip()
                    })
                    current_text = []
                
                # Set new category
                line_lower = line.lower()
                if 'key findings' in line_lower:
                    current_category = "Key Findings"
                elif 'business impact' in line_lower:
                    current_category = "Business Impact"
                elif 'future predictions' in line_lower:
                    current_category = "Future Predictions"
                elif 'actionable recommendations' in line_lower:
                    current_category = "Actionable Recommendations"
                continue
            
            # Check for insight markers and content
            if '[Insight' in line:
                # Save previous insight if exists
                if current_text:
                    insights.append({
                        'category': current_category,
                        'insight': ' '.join(current_text).strip()
                    })
                    current_text = []
                
                # Extract content after the marker
                parts = line.split(']', 1)
                if len(parts) > 1:
                    current_text = [parts[1].strip()]
            elif current_text and line:  # Only append non-empty lines
                current_text.append(line)
        
        # Add final insight if exists
        if current_text:
            insights.append({
                'category': current_category or 'Key Findings',
                'insight': ' '.join(current_text).strip()
            })
        
        # Clean and validate insights
        valid_insights = []
        for insight in insights:
            text = insight['insight'].strip()
            if text and len(text) > 10:  # Basic validation
                # Clean up any remaining markers or formatting
                text = text.replace('[Insight', '').replace(']', '').strip()
                valid_insights.append({
                    'category': insight['category'],
                    'insight': text
                })
        
        return valid_insights

    @st.cache_data(ttl=CACHE_TTL)
    def _clean_data(_self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataframe with optimized performance"""
        try:
            # Use more efficient operations
            df = df.copy()  # Avoid modifying original data
            
            # Process in batches for large datasets
            chunk_size = 10000
            if len(df) > chunk_size:
                return pd.concat([self._process_chunk(chunk) for chunk in np.array_split(df, len(df) // chunk_size + 1)])
            return _self._process_chunk(df)
        except Exception as e:
            st.error(f"Error cleaning data: {str(e)}")
            logger.error(f"Error cleaning data: {str(e)}")
            return df
            return df

    def _process_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a chunk of data efficiently with error handling"""
        try:
            # Create a copy to avoid modifying original data
            df = df.copy()
            
            # Process numeric and categorical columns separately
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            # Handle missing values efficiently
            if not numeric_cols.empty:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            if not categorical_cols.empty:
                for col in categorical_cols:
                    if df[col].isnull().any():
                        mode_val = df[col].mode()
                        df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown', inplace=True)
            
            # Remove duplicates after filling missing values
            df.drop_duplicates(inplace=True)
            
            # Optimize date parsing with parallel processing
            if not categorical_cols.empty:
                date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']
                
                def try_parse_dates(col):
                    try:
                        if df[col].dtype == 'object':
                            sample = df[col].dropna().iloc[0] if not df[col].empty else ''
                            if isinstance(sample, str):
                                for fmt in date_formats:
                                    try:
                                        return pd.to_datetime(df[col], format=fmt, errors='coerce')
                                    except (ValueError, TypeError):
                                        continue
                        return df[col]
                    except Exception as e:
                        logger.warning(f"Error parsing dates for column {col}: {str(e)}")
                        return df[col]
                
                # Process date columns in parallel with error handling
                try:
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        date_results = list(pool.map(try_parse_dates, categorical_cols))
                        for col, result in zip(categorical_cols, date_results):
                            df[col] = result
                except Exception as e:
                    logger.error(f"Error in parallel date processing: {str(e)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing data chunk: {str(e)}")
            raise

    def calculate_health_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data health score and statistics"""
        try:
            total_rows = len(df)
            total_columns = len(df.columns)
            
            if total_rows == 0 or total_columns == 0:
                return {
                    'health_score': 0,
                    'completeness': 0,
                    'duplicates': 0,
                    'column_types': {},
                    'row_count': 0,
                    'column_count': 0,
                    'memory_usage': 0
                }
            
            metrics = {
                'completeness': (1 - df.isnull().sum().sum() / (total_rows * total_columns)) * 100,
                'duplicates': (1 - df.duplicated().sum() / total_rows) * 100,
                'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'row_count': total_rows,
                'column_count': total_columns,
                'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            }
            
            # Calculate overall health score
            metrics['health_score'] = (metrics['completeness'] + metrics['duplicates']) / 2
            
            return metrics
        except Exception as e:
            st.error(f"Error calculating health score: {str(e)}")
            logger.error(f"Error calculating health score: {str(e)}")
            return None

    def _create_insight_prompt(self, summary: str, correlations: str) -> str:
        """Create a structured prompt for insight generation"""
        return f"""You are a data analyst tasked with extracting valuable insights from this dataset.
Analyze the data and provide clear, actionable insights in the following categories.

Dataset Summary:
{summary}

Correlations:
{correlations}

Format your insights exactly as follows:

1. Key Findings:
[Insight 1.1] Identify the most significant patterns and relationships in the data.
Focus on key metrics, their distributions, and correlations.
Include specific numbers and statistical evidence.

2. Business Impact:
[Insight 2.1] Evaluate how these findings affect business performance.
Quantify potential opportunities and risks.
Provide market context and competitive implications.

3. Future Predictions:
[Insight 3.1] Make data-driven forecasts about future trends.
Base predictions on current patterns and historical data.
Include confidence levels and potential timeline estimates.

4. Actionable Recommendations:
[Insight 4.1] Suggest specific actions based on the analysis.
Prioritize recommendations by potential impact.
Include implementation steps and expected outcomes.

Requirements for each insight:
1. Start with a clear, quantified observation
2. Support with specific data points from the analysis
3. Explain direct business implications
4. Provide concrete next steps or recommendations

Focus on quality over quantity. Each insight should be:
- Data-driven with specific numbers
- Clearly explained with supporting evidence
- Directly relevant to business objectives
- Immediately actionable

Use professional, concise language and ensure all insights are backed by the data provided."""

    async def _generate_insights_with_retry(self, prompt: str, progress_container) -> List[Dict[str, str]]:
        """Generate insights with retry logic"""
        max_retries = 2
        retry_delay = 1
        # Increase timeout and configure TCP settings
        timeout = aiohttp.ClientTimeout(total=180, connect=60, sock_read=60)
        connector = aiohttp.TCPConnector(limit=5, force_close=True, enable_cleanup_closed=True)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            for attempt in range(max_retries):
                try:
                    progress_container.text(f"Generating insights (attempt {attempt + 1}/{max_retries})...")
                    async with session.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "deepseek-chat",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.7,
                            "max_tokens": 1000,
                            "top_p": 0.95
                        },
                        raise_for_status=True
                    ) as response:
                        try:
                            result = await response.json()
                            response_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                            
                            if not response_text:
                                if attempt < max_retries - 1:
                                    progress_container.text("Empty response, retrying...")
                                    await asyncio.sleep(retry_delay * (attempt + 1))
                                    continue
                                return []
                            
                            insights = self._parse_insights(response_text)
                            if insights:
                                return insights
                            
                            if attempt < max_retries - 1:
                                progress_container.text("Invalid insights, retrying...")
                                await asyncio.sleep(retry_delay * (attempt + 1))
                                continue
                            
                        except (json.JSONDecodeError, asyncio.TimeoutError) as e:
                            if attempt < max_retries - 1:
                                progress_container.text(f"Error processing response: {str(e)}, retrying...")
                                await asyncio.sleep(retry_delay * (attempt + 1))
                                continue
                            raise
                            
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt < max_retries - 1:
                        progress_container.text(f"Connection error: {str(e)}, retrying...")
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    raise
                    
            return []
        
        return []

    def _validate_insights(self, insights: List[Dict[str, str]]) -> bool:
        """Validate insights meet quality requirements"""
        if not insights:
            return False

        # Ensure we have at least one insight per category
        required_categories = {
            "Key Findings",
            "Business Impact",
            "Future Predictions",
            "Actionable Recommendations"
        }

        # Get unique categories from insights
        categories = {insight["category"] for insight in insights}
        
        # Check if we have all required categories
        return required_categories.issubset(categories)

    async def generate_insights(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Generate AI-powered insights from the dataset with optimized performance"""
        if df.empty:
            return [{"category": "Error", "insight": "No data available for analysis."}]

        progress_container = None
        progress_bar = None

        try:
            # Initialize progress indicators
            progress_container = st.empty()
            progress_bar = st.progress(0)
            
            # Step 1: Data Analysis (20%)
            progress_container.text("Analyzing dataset...")
            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [
                        pool.submit(lambda: df.agg(['count', 'mean', 'std', 'min', 'max']).round(3).to_string()),
                        pool.submit(lambda: df.select_dtypes(['int64', 'float64']).corr().round(3).to_string())
                    ]
                    summary, correlations = [f.result() for f in futures]
            except Exception as e:
                logger.error(f"Error in data analysis: {str(e)}")
                raise ValueError("Failed to analyze dataset")
            
            progress_bar.progress(20)
            
            # Step 2: Cache Check (30%)
            progress_container.text("Checking analysis cache...")
            cache_key = f"insights_{hash(str(summary))}"
            if cache_key in st.session_state:
                return st.session_state[cache_key]
            
            progress_bar.progress(30)
            
            # Step 3: Prepare Analysis (40%)
            progress_container.text("Preparing analysis...")
            prompt = self._create_insight_prompt(summary, correlations)
            progress_bar.progress(40)
            
            # Step 4: Generate Insights (60%)
            progress_container.text("Generating insights...")
            insights = await self._generate_insights_with_retry(prompt, progress_container)
            if not insights:
                raise ValueError("Failed to generate insights")
            
            progress_bar.progress(60)
            
            # Step 5: Validate Results (80%)
            progress_container.text("Validating insights...")
            if not self._validate_insights(insights):
                raise ValueError("Generated insights did not meet quality requirements")
            
            progress_bar.progress(80)
            
            # Step 6: Cache Results (100%)
            progress_container.text("Finalizing analysis...")
            st.session_state[cache_key] = insights
            progress_bar.progress(100)
            
            st.success("Analysis completed successfully!")
            return insights

        except ValueError as e:
            logger.error(f"Validation error in insight generation: {str(e)}")
            return [{"category": "Error", "insight": str(e)}]
        except Exception as e:
            logger.error(f"Unexpected error in insight generation: {str(e)}", exc_info=True)
            return [{"category": "Error", "insight": "An unexpected error occurred during analysis."}]
        finally:
            if progress_container:
                progress_container.empty()
            if progress_bar:
                progress_bar.empty()

    def visualize_data(self, df: pd.DataFrame, chart_type: str, x_col: str, y_col: str,
                      color_col: Optional[str] = None) -> go.Figure:
        """Create interactive visualizations using Plotly"""
        try:
            if df.empty:
                st.error("No data available for visualization")
                return None
                
            if x_col not in df.columns or y_col not in df.columns:
                st.error("Selected columns not found in dataset")
                return None
                
            if color_col and color_col not in df.columns:
                st.error("Selected color column not found in dataset")
                return None
            
            # Set dark theme template
            template = go.layout.Template()
            template.layout.plot_bgcolor = '#1f2937'
            template.layout.paper_bgcolor = '#1f2937'
            template.layout.font.color = '#ffffff'
            
            if chart_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, template='plotly_dark')
            elif chart_type == "line":
                fig = px.line(df, x=x_col, y=y_col, color=color_col, template='plotly_dark')
            elif chart_type == "bar":
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, template='plotly_dark')
            elif chart_type == "histogram":
                fig = px.histogram(df, x=x_col, color=color_col, template='plotly_dark')
            elif chart_type == "box":
                fig = px.box(df, x=x_col, y=y_col, color=color_col, template='plotly_dark')
            else:
                st.error(f"Unsupported chart type: {chart_type}")
                return None
            
            fig.update_layout(
                title=f"{chart_type.title()} Plot: {y_col} vs {x_col}",
                xaxis_title=x_col,
                yaxis_title=y_col,
                showlegend=True,
                legend_title_text=color_col if color_col else "",
                plot_bgcolor='#1f2937',
                paper_bgcolor='#1f2937',
                font=dict(color='#ffffff')
            )
            
            return fig
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            logger.error(f"Error creating visualization: {str(e)}")
            return None

    async def generate_data_story(self, df: pd.DataFrame) -> str:
        """Generate a narrative story from the data analysis"""
        if df.empty:
            return "No data available for analysis."
            
        try:
            # Use a sample of the data for faster processing
            sample_size = min(1000, len(df))
            df_sample = df.sample(n=sample_size) if len(df) > sample_size else df
            
            summary = df_sample.describe().to_string()
            sample_data = df_sample.head().to_string()
            
            # Enhanced prompt for better storytelling
            prompt = f"""Create a concise data story focusing on key insights:

Summary Statistics:
{summary}

Sample Data:
{sample_data}

Structure:
1. Executive Summary (2-3 sentences)
2. Key Patterns (3-4 main points)
3. Business Impact (2-3 key implications)
4. Recommendations (2-3 actionable items)

Keep the analysis focused and actionable."""

            # Configure timeout and connection settings
            timeout = aiohttp.ClientTimeout(total=180, connect=60, sock_read=60)
            connector = aiohttp.TCPConnector(limit=5, force_close=True, enable_cleanup_closed=True)
            max_retries = 2
            retry_delay = 1

            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                        async with session.post(
                            "https://api.deepseek.com/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": "deepseek-chat",
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0.7,
                                "max_tokens": 1000,  # Reduced for faster response
                                "top_p": 0.95
                            }
                        ) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(retry_delay * (attempt + 1))
                                    continue
                                raise ValueError(f"API request failed: {error_text}")
                            
                            result = await response.json()
                            story = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                            
                            if story:
                                return story
                            
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay * (attempt + 1))
                                continue
                            
                            return "Unable to generate data story."
                            
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    raise
            
            return "Unable to generate data story after multiple attempts."
            
        except asyncio.TimeoutError:
            st.error("Request timed out. Please try again.")
            return "Request timed out. Please try again."
        except Exception as e:
            st.error(f"Error generating data story: {str(e)}")
            logger.error(f"Error generating data story: {str(e)}")
            return "Unable to generate data story at this time."