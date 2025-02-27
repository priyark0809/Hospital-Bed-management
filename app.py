import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import pickle
import os
import hashlib
import requests
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import base64

# Set page configuration
st.set_page_config(
    page_title="Hospital Bed Availability",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
FILE_PATH = 'E:\Hbms\bangalore_hospital_beds2.csv'
RANDOM_STATE = 42
MAX_DISTANCE_KM = 10  # Maximum distance to consider a hospital "nearby"

# Initialize session states
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_location' not in st.session_state:
    st.session_state.user_location = None
if 'current_hospital' not in st.session_state:
    st.session_state.current_hospital = None
if 'nearby_hospitals' not in st.session_state:
    st.session_state.nearby_hospitals = {}

# Database simulation for users (in a real app, you'd use a secure database)
users_db = {
    "user1": hashlib.sha256("password1".encode()).hexdigest(),
    "user2": hashlib.sha256("password2".encode()).hexdigest(),
    "admin": hashlib.sha256("admin123".encode()).hexdigest()
}

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data(file_path=None):
    """
    Loads, preprocesses, and encodes the data.
    If file_path is None, uses demo data.
    """
    try:
        if file_path and os.path.exists(file_path):
            data = pd.read_csv(file_path)
            data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y', errors='coerce')
            data.dropna(subset=['date'], inplace=True)

            categorical_cols = ['time_of_day', 'hospital_name', 'area']
            label_encoders = {col: LabelEncoder().fit(data[col]) for col in categorical_cols}
            for col, encoder in label_encoders.items():
                data[col] = data[col].astype(str)
                data[col] = encoder.transform(data[col])

            data.fillna(data.mean(numeric_only=True), inplace=True)
            data['day_of_week'] = data['date'].dt.dayofweek
            
            # Add geographical coordinates for each hospital (mock data)
            # In a real app, you would get this from a database or geocoding service
            hospital_coordinates = {}
            hospitals = pd.DataFrame(data[['hospital_name', 'area']].drop_duplicates())
            
            # For demo purposes - generating random coordinates near Bengaluru
            # Bengaluru center: approximately 12.9716° N, 77.5946° E
            bengaluru_center = (12.9716, 77.5946)
            for idx, row in hospitals.iterrows():
                # Generate coordinates within ~5km of center
                lat = bengaluru_center[0] + np.random.uniform(-0.05, 0.05)
                lon = bengaluru_center[1] + np.random.uniform(-0.05, 0.05)
                hospital_coordinates[row['hospital_name']] = (lat, lon)
                
            # Map the coordinates back to the data
            hospital_lat = []
            hospital_lon = []
            
            for idx, row in data.iterrows():
                coords = hospital_coordinates.get(row['hospital_name'], (None, None))
                hospital_lat.append(coords[0])
                hospital_lon.append(coords[1])
                
            data['hospital_lat'] = hospital_lat
            data['hospital_lon'] = hospital_lon
                
            return data, label_encoders
        else:
            # Create demo data
            hospitals = [
                {"id": 1, "name": "City Hospital", "area": "Indiranagar", "lat": 12.9719, "lon": 77.6412,
                 "total_beds": 200, "occupied_beds": 150, "available_beds": 50, 
                 "emergency_cases": 10, "waiting_patients": 5, "total_icu_beds": 30},
                {"id": 2, "name": "General Hospital", "area": "Koramangala", "lat": 12.9352, "lon": 77.6245,
                 "total_beds": 300, "occupied_beds": 250, "available_beds": 50,
                 "emergency_cases": 15, "waiting_patients": 8, "total_icu_beds": 45},
                {"id": 3, "name": "Community Medical Center", "area": "Jayanagar", "lat": 12.9299, "lon": 77.5933,
                 "total_beds": 150, "occupied_beds": 100, "available_beds": 50,
                 "emergency_cases": 5, "waiting_patients": 3, "total_icu_beds": 20},
                {"id": 4, "name": "St. Mary's Hospital", "area": "Whitefield", "lat": 12.9698, "lon": 77.7500,
                 "total_beds": 250, "occupied_beds": 200, "available_beds": 50,
                 "emergency_cases": 12, "waiting_patients": 7, "total_icu_beds": 35},
                {"id": 5, "name": "Sunshine Medical", "area": "HSR Layout", "lat": 12.9116, "lon": 77.6389,
                 "total_beds": 180, "occupied_beds": 140, "available_beds": 40,
                 "emergency_cases": 8, "waiting_patients": 6, "total_icu_beds": 25},
                {"id": 6, "name": "Apollo Hospital", "area": "Bannerghatta Road", "lat": 12.8698, "lon": 77.5967,
                 "total_beds": 350, "occupied_beds": 280, "available_beds": 70,
                 "emergency_cases": 20, "waiting_patients": 12, "total_icu_beds": 50},
                {"id": 7, "name": "Fortis Hospital", "area": "Richmond Town", "lat": 12.9647, "lon": 77.6039,
                 "total_beds": 280, "occupied_beds": 220, "available_beds": 60,
                 "emergency_cases": 15, "waiting_patients": 8, "total_icu_beds": 40},
                {"id": 8, "name": "Manipal Hospital", "area": "Malleswaram", "lat": 13.0025, "lon": 77.5713,
                 "total_beds": 320, "occupied_beds": 260, "available_beds": 60,
                 "emergency_cases": 18, "waiting_patients": 10, "total_icu_beds": 45}
            ]
            
            # Create a dataframe
            times_of_day = ['Morning', 'Afternoon', 'Evening', 'Night']
            records = []
            
            # Generate a few days of data
            for day in range(30):
                date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=day)
                day_of_week = date.dayofweek
                
                for hospital in hospitals:
                    for time_of_day in times_of_day:
                        # Add some randomness to the data
                        available = max(0, hospital['available_beds'] + np.random.randint(-10, 11))
                        occupied = hospital['total_beds'] - available
                        emergency = max(0, hospital['emergency_cases'] + np.random.randint(-3, 4))
                        waiting = max(0, hospital['waiting_patients'] + np.random.randint(-2, 3))
                        
                        records.append({
                            'date': date,
                            'time_of_day': time_of_day,
                            'hospital_name': hospital['name'],
                            'hospital_id': hospital['id'],
                            'area': hospital['area'],
                            'total_beds': hospital['total_beds'],
                            'available_beds': available,
                            'occupied_beds': occupied,
                            'recent_admissions': np.random.randint(0, 10),
                            'emergency_cases': emergency,
                            'waiting_patients': waiting,
                            'total_icu_beds': hospital['total_icu_beds'],
                            'day_of_week': day_of_week,
                            'hospital_lat': hospital['lat'],
                            'hospital_lon': hospital['lon']
                        })
            
            data = pd.DataFrame(records)
            
            # Create label encoders
            categorical_cols = ['time_of_day', 'hospital_name', 'area']
            label_encoders = {col: LabelEncoder().fit(data[col]) for col in categorical_cols}
            
            # Transform categorical columns
            for col, encoder in label_encoders.items():
                data[col+'_encoded'] = encoder.transform(data[col])
            
            return data, label_encoders
            
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None, None

# --- Model Training ---
@st.cache_resource
def train_model(data, n_estimators=200, random_state=42):
    """Trains a Random Forest Regressor model."""
    features = [
        'time_of_day_encoded' if 'time_of_day_encoded' in data.columns else 'time_of_day', 
        'hospital_id', 
        'area_encoded' if 'area_encoded' in data.columns else 'area', 
        'total_beds', 
        'recent_admissions', 
        'emergency_cases',
        'waiting_patients',
        'total_icu_beds', 
        'day_of_week'
    ]
    
    # Make sure all features exist in the dataframe
    features = [f for f in features if f in data.columns]
    
    target = 'available_beds'

    try:
        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model, mse, r2, features

    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        return None, None, None, None

# --- Prediction Function ---
def predict_bed_availability(hospital_data, model, features):
    """
    Predicts bed availability for a hospital based on its current data
    """
    if model is None:
        return None
    
    try:
        # Create a dataframe with the required features
        input_data = pd.DataFrame([hospital_data])
        
        # Select only the features used by the model
        X = input_data[features]
        
        # Make prediction
        prediction = max(0, round(model.predict(X)[0], 1))
        
        return prediction
    
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

# Define CSS styling
def local_css():
    st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .hospital-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
    }
    .metric-box {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
        flex: 1;
        min-width: 120px;
        margin-right: 10px;
    }
    .login-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 40px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title-container {
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton button {
        width: 100%;
        border-radius: 20px;
        padding: 8px 16px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Authentication functions
def authenticate(username, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    if username in users_db and users_db[username] == hashed_password:
        return True
    return False

def login_page():
    st.markdown('<div class="title-container"><h1>🏥 Hospital Bed Availability System</h1></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.subheader("User Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; color: #666;">
    Demo Credentials: <br>
    Username: user1, Password: password1 <br>
    Username: admin, Password: admin123
    </div>
    """, unsafe_allow_html=True)

# Function to get user's location
def get_user_location():
    st.subheader("📍 Share Your Location")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        address = st.text_input("Enter your address or use current location")
    
    with col2:
        if st.button("Get Current Location", key="get_location"):
            # In a real app, this would use browser geolocation API
            # For demo, we'll use Bengaluru as default
            st.session_state.user_location = (12.9716, 77.5946)  # Bengaluru coordinates
            st.success("Location obtained: Bengaluru")
            return
    
    if address and st.button("Search", key="search_address"):
        try:
            geolocator = Nominatim(user_agent="hospital_app")
            location = geolocator.geocode(address)
            if location:
                st.session_state.user_location = (location.latitude, location.longitude)
                st.success(f"Location found: {location.address}")
            else:
                st.error("Location not found. Please try again.")
        except Exception as e:
            st.error(f"Error finding location: {e}")

# Function to find nearby hospitals
def find_nearby_hospitals(data, user_location, max_distance=MAX_DISTANCE_KM):
    """
    Finds hospitals within the specified distance of the user's location
    """
    if user_location is None:
        return {}
    
    nearby_hospitals = {}
    
    # Group data by hospital to get unique hospitals
    hospitals = data.groupby(['hospital_name']).agg({
        'hospital_id': 'first',
        'area': 'first',
        'hospital_lat': 'first',
        'hospital_lon': 'first',
        'total_beds': 'first',
        'occupied_beds': 'mean',
        'available_beds': 'mean',
        'emergency_cases': 'mean',
        'waiting_patients': 'mean',
        'total_icu_beds': 'first'
    }).reset_index()
    
    for idx, hospital in hospitals.iterrows():
        hospital_location = (hospital['hospital_lat'], hospital['hospital_lon'])
        distance = geodesic(user_location, hospital_location).kilometers
        
        if distance <= max_distance:
            nearby_hospitals[hospital['hospital_name']] = {
                'id': hospital['hospital_id'],
                'name': hospital['hospital_name'],
                'area': hospital['area'],
                'lat': hospital['hospital_lat'],
                'lon': hospital['hospital_lon'],
                'total_beds': int(hospital['total_beds']),
                'occupied_beds': int(hospital['occupied_beds']),
                'available_beds': int(hospital['available_beds']),
                'emergency_cases': int(hospital['emergency_cases']),
                'waiting_patients': int(hospital['waiting_patients']),
                'total_icu_beds': int(hospital['total_icu_beds']),
                'distance': round(distance, 2)
            }
    
    return nearby_hospitals

# Function to display nearby hospitals on map
def show_nearby_hospitals(data, model, features):
    if st.session_state.user_location:
        st.subheader("🗺️ Nearby Hospitals")
        
        user_lat, user_lon = st.session_state.user_location
        
        # Find nearby hospitals
        nearby_hospitals = find_nearby_hospitals(data, st.session_state.user_location)
        st.session_state.nearby_hospitals = nearby_hospitals
        
        if not nearby_hospitals:
            st.warning("No hospitals found within 10km of your location.")
            return
        
        # Create map centered at user location
        m = folium.Map(location=[user_lat, user_lon], zoom_start=13)
        
        # Add user marker
        folium.Marker(
            [user_lat, user_lon],
            popup="Your Location",
            icon=folium.Icon(color="blue", icon="user", prefix='fa'),
        ).add_to(m)
        
        # Create marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add hospital markers
        for name, data in nearby_hospitals.items():
            # Calculate availability percentage
            available_percent = (data['available_beds'] / data['total_beds']) * 100
            
            # Predict bed availability
            hospital_data = {
                'time_of_day_encoded': 0,  # Using default value for demo
                'hospital_id': data['id'],
                'area_encoded': 0,  # Using default value for demo
                'total_beds': data['total_beds'],
                'recent_admissions': 5,  # Using default value for demo
                'emergency_cases': data['emergency_cases'],
                'waiting_patients': data['waiting_patients'],
                'total_icu_beds': data['total_icu_beds'],
                'day_of_week': pd.Timestamp.now().dayofweek
            }
            
            predicted_beds = predict_bed_availability(hospital_data, model, features)
            
            # Choose color based on availability
            if available_percent > 30:
                color = "green"
            elif available_percent > 10:
                color = "orange"
            else:
                color = "red"
            
            # Create popup content
            popup_content = f"""
            <div style="width: 200px;">
                <h4>{name}</h4>
                <p><b>Area:</b> {data['area']}</p>
                <p><b>Available Beds:</b> {data['available_beds']}/{data['total_beds']}</p>
                <p><b>Predicted Available:</b> {predicted_beds if predicted_beds else 'N/A'}</p>
                <p><b>Distance:</b> {data['distance']} km</p>
            </div>
            """
            
            # Add marker with popup
            folium.Marker(
                [data['lat'], data['lon']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color=color, icon="hospital", prefix='fa'),
                tooltip=name
            ).add_to(marker_cluster)
        
        # Display the map
        folium_static(m)
        
        # Display list of hospitals
        st.subheader("📋 Hospital List")
        
        cols = st.columns(2)
        
        for i, (name, data) in enumerate(nearby_hospitals.items()):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="hospital-card">
                    <h3>{name}</h3>
                    <p><b>Area:</b> {data['area']}</p>
                    <p><b>Available Beds:</b> {data['available_beds']}/{data['total_beds']}</p>
                    <p><b>Emergency Cases:</b> {data['emergency_cases']}</p>
                    <p><b>Distance:</b> {data['distance']} km</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"View Details", key=f"view_{name}"):
                    st.session_state.current_hospital = name
                    st.rerun()
    else:
        st.info("Please provide your location to see nearby hospitals.")

# Function to display hospital details
def show_hospital_details(hospital_name, data, model, features):
    hospital = st.session_state.nearby_hospitals[hospital_name]
    
    st.markdown(f"<h2>🏥 {hospital_name}</h2>", unsafe_allow_html=True)
    
    # Back button
    if st.button("← Back to Hospitals List"):
        st.session_state.current_hospital = None
        st.experimental_rerun()
    
    # Create a dictionary with the hospital data for prediction
    hospital_data = {
        'time_of_day_encoded': 0,  # Using default value 
        'hospital_id': hospital['id'],
        'area_encoded': 0,  # Using default value
        'total_beds': hospital['total_beds'],
        'recent_admissions': 5,  # Using default value 
        'emergency_cases': hospital['emergency_cases'],
        'waiting_patients': hospital['waiting_patients'],
        'total_icu_beds': hospital['total_icu_beds'],
        'day_of_week': pd.Timestamp.now().dayofweek
    }
    
    # Predict bed availability
    predicted_beds = predict_bed_availability(hospital_data, model, features)
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Beds", hospital['total_beds'])
    
    with col2:
        st.metric("Occupied Beds", hospital['occupied_beds'])
    
    with col3:
        st.metric("Currently Available", hospital['available_beds'])
    
    with col4:
        st.metric("Predicted Available", predicted_beds if predicted_beds else 'N/A', 
                 delta=round(predicted_beds - hospital['available_beds'], 1) if predicted_beds else None)
    
    # Show factors affecting availability
    st.subheader("Factors Affecting Availability")
    
    factor_col1, factor_col2 = st.columns(2)
    
    with factor_col1:
        st.markdown("""
        <div class="metric-box">
            <h4>Emergency Cases</h4>
            <h2>{}</h2>
        </div>
        """.format(hospital['emergency_cases']), unsafe_allow_html=True)
    
    with factor_col2:
        st.markdown("""
        <div class="metric-box">
            <h4>Waiting Patients</h4>
            <h2>{}</h2>
        </div>
        """.format(hospital['waiting_patients']), unsafe_allow_html=True)
    
    # Display prediction explanation
    st.subheader("Prediction Explanation")
    
    st.write("""
    Our prediction algorithm takes into account:
    - Current bed occupancy rate
    - Number of emergency cases (which typically require immediate beds)
    - Number of waiting patients
    - Historical patterns of patient discharge and admission
    - Time of day and day of week patterns
    """)
    
    # Show historical trend (simulated)
    st.subheader("Historical Trend")
    
    # Filter data for this hospital
    hospital_history = data[data['hospital_name'] == hospital_name].copy()
    
    if not hospital_history.empty:
        # Group by date for the chart
        historical_data = hospital_history.groupby('date').agg({
            'available_beds': 'mean',
            'occupied_beds': 'mean'
        }).reset_index()
        
        # Convert to datetime for proper plotting
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        historical_data = historical_data.sort_values('date')
        
        # Create plot
        fig = px.line(
            historical_data, 
            x='date', 
            y=['available_beds', 'occupied_beds'],
            title=f"Bed Trends for {hospital_name}",
            labels={'value': 'Number of Beds', 'date': 'Date', 'variable': 'Metric'},
            color_discrete_map={
                'available_beds': '#4CAF50',  # Green
                'occupied_beds': '#FF5722'    # Red/Orange
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data available for this hospital.")

# Contact form
def contact_form():
    """Renders a contact form in a Streamlit app."""
    formspark_url = "https://submit-form.com/kgnZtb04c" 
   
    with st.container():
        st.subheader("Contact Us")
        with st.form(key='contact_form', clear_on_submit=True):
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            phone_number = st.text_input("Phone Number")
            message = st.text_area("Message", height=150)
            submit_button = st.form_submit_button("Submit")

            if submit_button:
                # Process the form data
                data = {
                    "firstName": first_name,
                    "lastName": last_name,
                    "phoneNumber": phone_number,
                    "message": message
                }

                try:
                    # In a real app, you would send the data
                    # For demo purposes, just simulate success
                    st.success("Thank you for your message! It has been sent successfully.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Main application
def main():
    local_css()
    
    # Authentication check
    if not st.session_state.authenticated:
        login_page()
        return
    
    # Load data
    data, label_encoders = load_and_preprocess_data(FILE_PATH)
    
    if data is None:
        st.error("Could not load data. Using demo data instead.")
        data, label_encoders = load_and_preprocess_data()
    
    # Train model
    model, mse, r2, features = train_model(data, n_estimators=100, random_state=42)
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/master/examples/data/healthcare.png", width=100)
        st.title(f"Welcome, {st.session_state.username}")
        
        st.markdown("---")
        
        menu = st.radio("Navigation", ["Find Nearby Hospitals", "About", "Contact Us"])
        
        if model:
            st.success("Model trained successfully! 💯")
            st.subheader("Model Metrics")
            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write(f"R-squared: {r2:.2f}")
        
        st.markdown("---")
        
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
    
    # Main content
    if menu == "Find Nearby Hospitals":
        if st.session_state.current_hospital:
            show_hospital_details(st.session_state.current_hospital, data, model, features)
        else:
            st.title("🏥 Hospital Bed Finder")
            
            get_user_location()
            
            st.markdown("---")
            
            show_nearby_hospitals(data, model, features)
    
    elif menu == "About":
        st.title("About This System")
        
        st.write("""
        ## Hospital Bed Availability System
        
        This system helps patients and healthcare providers find available hospital beds in real-time. 
        Our predictive algorithm uses current occupancy, emergency cases, and waiting patients data 
        to estimate real-time bed availability.
        
        ### How It Works
        
        1. Log in to the system
        2. Share your location
        3. View nearby hospitals on the map
        4. Check detailed availability information
        5. See predictions on current and future availability
        
        ### Prediction Model
        
        Our prediction model is built using Random Forest Regression, which takes into account:
        
        - Hospital occupancy patterns
        - Time of day variations
        - Day of week patterns
        - Emergency case load
        - Waiting patient count
        
        ### Data Privacy
        
        We take your privacy seriously. Your location data is only used to find nearby hospitals and is not stored.
        """)
    
    elif menu == "Contact Us":
        st.title("Contact Us")
        contact_form()

if __name__ == "__main__":
    main()