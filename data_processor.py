import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import streamlit as st
import asyncio
from pathlib import Path
import json
import aiohttp
import logging
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, CACHE_TTL, logger

class DataProcessor:
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

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataframe"""
        try:
            # Remove duplicate rows
            df = df.drop_duplicates()
            
            # Handle missing values
            df = df.fillna({
                col: df[col].mode()[0] if df[col].dtype == 'object' else 
                df[col].mean() if df[col].dtype in ['int64', 'float64'] else 
                df[col].fillna('Unknown')
                for col in df.columns
            })
            
            # Convert date columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to detect if column contains dates
                    date_sample = df[col].dropna().iloc[0] if not df[col].empty else ''
                    if isinstance(date_sample, str):
                        try:
                            # Common date formats to try
                            formats = [
                                '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y',
                                '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S',
                                '%m/%d/%Y %H:%M:%S'
                            ]
                            for fmt in formats:
                                try:
                                    df[col] = pd.to_datetime(df[col], format=fmt)
                                    break
                                except ValueError:
                                    continue
                        except Exception as e:
                            logger.warning(f"Failed to parse dates in column {col}: {str(e)}")
            
            return df
        except Exception as e:
            st.error(f"Error cleaning data: {str(e)}")
            logger.error(f"Error cleaning data: {str(e)}")
            return df

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

    async def generate_insights(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Generate AI-powered insights from the dataset"""
        if df.empty:
            return [{"insight": "No data available for analysis."}]
            
        try:
            summary = df.describe().to_string()
            correlations = df.select_dtypes(['int64', 'float64']).corr().to_string()
            
            prompt = f"""Analyze this dataset and provide 5 key insights in a clear, structured format.

Summary Statistics:
{summary}

Correlations:
{correlations}

Please provide insights in this format:
1. [First insight about the data]
2. [Second insight focusing on key patterns]
3. [Third insight about relationships]
4. [Fourth insight about distributions]
5. [Fifth insight about notable findings]

Make each insight clear, specific, and data-driven."""
            
            async with aiohttp.ClientSession() as session:
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
                    timeout=60
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"API request failed: {error_text}")
                        
                    result = await response.json()
                    response_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    if not response_text:
                        return [{"insight": "No insights could be generated. Please try again."}]
                    
                    # Clean and format insights
                    insights = []
                    for line in response_text.split('\n'):
                        line = line.strip()
                        if line and any(line.startswith(str(i)) for i in range(1, 6)):
                            # Remove number prefix and any common separators
                            insight = line.split('.', 1)[-1].strip()
                            if insight:
                                insights.append({'insight': insight})
                    
                    return insights if insights else [{"insight": "No clear insights could be extracted. Please try again."}]
        except asyncio.TimeoutError:
            st.error("Request timed out. Please try again.")
            return [{"insight": "Request timed out. Please try again."}]
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
            logger.error(f"Error generating insights: {str(e)}")
            return [{"insight": "Failed to generate insights. Please try again later."}]

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
            summary = df.describe().to_string()
            sample_data = df.head().to_string()
            
            prompt = f"""Create a comprehensive data story from this dataset:
            Summary Statistics:
            {summary}
            
            Sample Data:
            {sample_data}
            
            Create a narrative that explains the key trends, patterns, and insights in a clear, engaging way."""
            
            async with aiohttp.ClientSession() as session:
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
                        "max_tokens": 1500,
                        "top_p": 0.95
                    },
                    timeout=30
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"API request failed: {error_text}")
                        
                    result = await response.json()
                    return result.get('choices', [{}])[0].get('message', {}).get('content', 'Unable to generate data story.')
        except asyncio.TimeoutError:
            st.error("Request timed out. Please try again.")
            return "Request timed out. Please try again."
        except Exception as e:
            st.error(f"Error generating data story: {str(e)}")
            logger.error(f"Error generating data story: {str(e)}")
            return "Unable to generate data story at this time."