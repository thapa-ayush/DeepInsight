import pytest
import pandas as pd
from data_processor import DataProcessor
from pathlib import Path

@pytest.fixture
def data_processor():
    return DataProcessor("dummy-api-key")

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, 2, 3, None, 5],
        'B': ['x', 'y', None, 'w', 'z'],
        'C': [1.1, 2.2, 3.3, 4.4, None]
    })

def test_clean_data(data_processor, sample_df):
    cleaned_df = data_processor._clean_data(sample_df)
    assert cleaned_df is not None
    assert cleaned_df.isnull().sum().sum() == 0

def test_calculate_health_score(data_processor, sample_df):
    health_metrics = data_processor.calculate_health_score(sample_df)
    assert 'health_score' in health_metrics
    assert 'completeness' in health_metrics
    assert 'duplicates' in health_metrics
    assert isinstance(health_metrics['health_score'], float)
    assert 0 <= health_metrics['health_score'] <= 100

def test_visualize_data(data_processor, sample_df):
    fig = data_processor.visualize_data(sample_df, "scatter", "A", "C")
    assert fig is not None
    assert fig.layout.title.text == "Scatter Plot: C vs A"