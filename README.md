# DeepInsight Data Analysis Tool

A data analysis tool I built to help analyze and visualize datasets using Python and Streamlit. This project was created as part of my learning journey in data science and web development.

## Features

- Data Upload and Processing

  - Support for CSV, Excel, Parquet, and JSON files
  - Automatic data cleaning and type inference
  - Basic data quality checks

- Analysis Tools

  - Statistical summaries
  - Data health scoring
  - Missing value analysis
  - Column-wise statistics

- Interactive Visualizations

  - Scatter plots
  - Line charts
  - Bar graphs
  - Histograms
  - Box plots
  - Correlation heatmaps

- AI-Powered Analysis
  - Pattern detection
  - Trend analysis
  - Automated insights generation
  - Natural language data stories

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/thapa-ayush/DeepInsight.git
   cd DeepInsight
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   - Copy `.env.example` to `.env`
   - Add your DeepSeek API key to enable AI features

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main application file with Streamlit UI
- `data_processor.py`: Core data processing and analysis logic
- `config.py`: Configuration settings and constants
- `auth.py`: Authentication and API key management
- `tests/`: Unit tests
- `.github/workflows/`: CI/CD pipeline configuration

## Development Setup

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run tests:
   ```bash
   pytest
   ```

## Contributing

If you'd like to contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Credits

Built by Ayush Thapa as a learning project to explore data analysis, visualization, and AI integration. The project uses several open-source libraries:

- Streamlit for the web interface
- Pandas for data processing
- Plotly for interactive visualizations
- DeepSeek API for AI analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.
