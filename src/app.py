import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_fetcher import DataFetcher

# Page configuration
st.set_page_config(
    page_title="Balloon & Aircraft Tracker",
    page_icon="🎈✈️",
    layout="wide"
)

# Initialize the data fetcher
@st.cache_resource
def get_data_fetcher():
    return DataFetcher()

data_fetcher = get_data_fetcher()

# Title and description
st.title("🎈✈️ Balloon & Aircraft Tracker")
st.markdown("""
This application tracks WindBorne's balloon constellation and nearby aircraft using data from:
- WindBorne Systems API (balloon data)
- OpenSky Network API (aircraft data)

The map updates every 5 minutes to show the latest positions of balloons and aircraft.
""")

# Fetch current data
with st.spinner("Fetching latest data..."):
    data = data_fetcher.get_current_data()
    stats = data_fetcher.get_statistics()
    
    # Debug information
    st.sidebar.write("### Debug Information")
    if not data.empty:
        st.sidebar.write(f"Total records: {len(data)}")
        st.sidebar.write("Data types:")
        st.sidebar.write(data['type'].value_counts())
        st.sidebar.write("\nSample balloon data:")
        balloon_sample = data[data['type'] == 'balloon'].head(2)
        if not balloon_sample.empty:
            st.sidebar.write(balloon_sample)
        else:
            st.sidebar.write("No balloon data found")
    else:
        st.sidebar.write("No data received")

# Display statistics in columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Balloons", stats.get('total_balloons', 0))
with col2:
    st.metric("Total Aircraft", stats.get('total_aircraft', 0))
with col3:
    st.metric("Avg. Balloon Altitude (m)", f"{stats.get('avg_balloon_altitude', 0):,.0f}")
with col4:
    st.metric("Avg. Aircraft Altitude (m)", f"{stats.get('avg_aircraft_altitude', 0):,.0f}")

# Last update time
st.caption(f"Last updated: {stats.get('last_update', 'Unknown')}")

# Filters
st.sidebar.title("Filters")
show_balloons = st.sidebar.checkbox("Show Balloons", value=True)
show_aircraft = st.sidebar.checkbox("Show Aircraft", value=True)
min_altitude = st.sidebar.slider("Minimum Altitude (m)", 0, 20000, 0)
max_altitude = st.sidebar.slider("Maximum Altitude (m)", 0, 20000, 20000)

# Filter data based on user selection
filtered_data = data[
    (data['altitude'].between(min_altitude, max_altitude)) &
    ((data['type'] == 'balloon') & show_balloons |
     (data['type'] == 'aircraft') & show_aircraft)
]

# Create the map
fig = go.Figure()

if not filtered_data.empty:
    # Add balloons
    balloon_data = filtered_data[filtered_data['type'] == 'balloon']
    if not balloon_data.empty:
        fig.add_trace(go.Scattergeo(
            lon=balloon_data['longitude'],
            lat=balloon_data['latitude'],
            text=balloon_data.apply(lambda row: f"Balloon {row['id']}<br>Alt: {row['altitude']:,.0f}m", axis=1),
            mode='markers',
            marker=dict(
                size=10,
                symbol='circle',
                color='red',
                line=dict(width=1, color='white')
            ),
            name='Balloons'
        ))

    # Add aircraft
    aircraft_data = filtered_data[filtered_data['type'] == 'aircraft']
    if not aircraft_data.empty:
        # Create a custom symbol for aircraft based on heading
        fig.add_trace(go.Scattergeo(
            lon=aircraft_data['longitude'],
            lat=aircraft_data['latitude'],
            text=aircraft_data.apply(lambda row: 
                f"Aircraft {row['id']}<br>" +
                f"Alt: {row['altitude']:,.0f}m<br>" +
                f"Speed: {row['velocity']:,.0f}m/s<br>" +
                f"Heading: {row['heading']:,.1f}°<br>" +
                f"Vertical Rate: {row['vertical_rate']:,.1f}m/s", axis=1),
            mode='markers',
            marker=dict(
                size=8,
                symbol='triangle-up',  # Changed from 'airplane' to 'triangle-up'
                color='blue',
                line=dict(width=1, color='white')
            ),
            name='Aircraft'
        ))

    # Update map layout
    fig.update_layout(
        title='Balloon and Aircraft Positions',
        showlegend=True,
        geo=dict(
            showland=True,
            showcountries=True,
            showocean=True,
            countrywidth=0.5,
            landcolor='rgb(243, 243, 243)',
            oceancolor='rgb(204, 229, 255)',
            projection_type='equirectangular',
            center=dict(
                lat=filtered_data['latitude'].mean(),
                lon=filtered_data['longitude'].mean()
            ),
            projection_scale=1.5
        ),
        height=700
    )

    # Display the map
    st.plotly_chart(fig, use_container_width=True)

    # Display data tables
    if show_balloons and not balloon_data.empty:
        st.subheader("Balloon Data")
        st.dataframe(
            balloon_data[['id', 'latitude', 'longitude', 'altitude', 'timestamp']]
            .sort_values('timestamp', ascending=False)
        )

    if show_aircraft and not aircraft_data.empty:
        st.subheader("Aircraft Data")
        st.dataframe(
            aircraft_data[[
                'id', 'latitude', 'longitude', 'altitude',
                'velocity', 'heading', 'vertical_rate', 'timestamp'
            ]].sort_values('timestamp', ascending=False)
        )
else:
    st.warning("No data available for the selected filters.") 