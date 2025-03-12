# WindBorne Constellation Analyzer

This application provides real-time analysis and visualization of WindBorne Systems' global balloon constellation data. It fetches and processes 24 hours of historical balloon position data to provide operational insights and trajectory analysis.

## Features

- Real-time data fetching from WindBorne's constellation API
- Interactive 3D visualization of balloon trajectories
- Analysis of balloon movement patterns and environmental conditions
- Automatic updates with the latest 24 hours of data

## Setup

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run src/app.py
   ```

## Project Structure

- `src/app.py`: Main Streamlit application
- `src/data_fetcher.py`: API data fetching and processing
- `src/analysis.py`: Data analysis and insights generation
- `static/`: Static assets and cached data

## Notes

This project focuses on providing operational insights by analyzing balloon trajectories and their interaction with atmospheric conditions. The visualization and analysis tools help operators understand constellation behavior and optimize deployment strategies.