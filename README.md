# 🎈✈️ Balloon & Aircraft Tracker

A real-time visualization tool that combines WindBorne Systems' balloon constellation data with OpenSky Network's aircraft tracking data to provide insights into aerial traffic and potential interactions.

## Features

- **Real-time Tracking**: Displays current positions of both balloons and aircraft
- **Interactive Map**: 
  - Global visualization with markers for both balloons and aircraft
  - Color-coded markers (red for balloons, blue for aircraft)
  - Detailed tooltips showing altitude, speed, and heading information
- **Filtering Capabilities**:
  - Toggle visibility of balloons and aircraft independently
  - Filter by altitude range
  - Automatic updates every 5 minutes
- **Data Tables**:
  - Detailed balloon position information
  - Comprehensive aircraft data including velocity and vertical rates
- **Statistics Dashboard**:
  - Total count of tracked balloons and aircraft
  - Average altitudes for both types of vehicles
  - Last update timestamp

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd WindBorneChallenge
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you're in the project directory and your virtual environment is activated

2. Run the Streamlit application:
```bash
streamlit run src/app.py
```

3. The application will open in your default web browser. If it doesn't, navigate to the URL shown in the terminal (typically `http://localhost:8501`)

## Data Sources

- **WindBorne Systems API**: Provides balloon constellation data
  - Updates every hour
  - Includes position and altitude information
  - 24-hour historical data available

- **OpenSky Network API**: Provides real-time aircraft data
  - Free, public aircraft tracking data
  - Includes position, altitude, velocity, and heading information
  - No API key required

## Technical Details

- **Backend**: Python with Streamlit framework
- **Data Processing**: Pandas for data manipulation
- **Visualization**: Plotly for interactive maps and charts
- **APIs**: 
  - REST API calls to WindBorne Systems
  - OpenSky Network API integration
- **Caching**: Implemented to reduce API calls and improve performance

## Project Structure

```
WindBorneChallenge/
├── src/
│   ├── app.py              # Main Streamlit application
│   └── data_fetcher.py     # Data fetching and processing logic
├── venv/                   # Virtual environment (generated)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Dependencies

- Python 3.8+
- streamlit
- pandas
- plotly
- requests
- numpy

## Performance Considerations

- Data is cached for 5 minutes to reduce API load
- Aircraft data is fetched only for regions where balloons are present
- Efficient data processing with vectorized operations
- Automatic cleanup of old cache entries

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[Your chosen license]

## Acknowledgments

- WindBorne Systems for providing balloon constellation data
- OpenSky Network for their free aircraft tracking API
- Streamlit and Plotly for their excellent visualization tools