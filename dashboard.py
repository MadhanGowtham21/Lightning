import streamlit as st
import requests
import pandas as pd
import numpy as np
import pickle
import os
from twilio.rest import Client

# ---------- Configuration ----------
CHANNEL_ID = "2885781"
READ_API_KEY = "6J1406GVYHSB1OJC"
FIELDS = ["Temperature", "Humidity", "Rain", "Flame", "Waterlevel", "Sound", "Voltage", "Current"]

# Twilio Configuration
TWILIO_PHONE_NUMBER = "whatsapp:+14155238886"  # Your Twilio WhatsApp number
AUTH_TOKEN = "1602c5b71508bcc28efc1e79657f2ea4"  # Your Twilio Auth Token
SID = "AC3d305967e3e0b493a29623a00409c805"  # Your Twilio SID
client = Client(SID, AUTH_TOKEN)


# ---------- Check Login Credentials ----------
def check_login(username, password):
    correct_username = "Lightning"
    correct_password = "Light@1234"
    return username == correct_username and password == correct_password


# ---------- Fetch Data from ThingSpeak ----------
def fetch_data():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=100"
    response = requests.get(url)
    if response.status_code == 200:
        feeds = response.json()['feeds']
        df = pd.DataFrame(feeds)
        df['created_at'] = pd.to_datetime(df['created_at'])

        for i in range(8):
            df[f'field{i + 1}'] = pd.to_numeric(df[f'field{i + 1}'], errors='coerce')
            df.rename(columns={f'field{i + 1}': FIELDS[i]}, inplace=True)

        return df[['created_at'] + FIELDS]
    else:
        st.error("Failed to fetch data from ThingSpeak")
        return pd.DataFrame()


# ---------- Load ML Model ----------
def load_model():
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    else:
        st.warning("ML model not found. Please train the model and place 'model.pkl' in the same folder.")
        return None


# ---------- Send WhatsApp Message Function ----------
def send_whatsapp_alert(to, message):
    try:
        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to
        )
        st.success(f"Alert sent to {to}: {message}")
    except Exception as e:
        st.error(f"Failed to send message: {e}")


# ---------- Streamlit App ----------
st.set_page_config(page_title="Firework Dashboard", layout="wide")

# ---------- Login Page ----------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Login")
    st.write("Please enter your credentials to access the dashboard.")

    # Login form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Incorrect username or password. Please try again.")

# ---------- Main Dashboard after Login ----------
if st.session_state.logged_in:
    page = st.sidebar.radio("📁 Pages", ["Home", "Sensors", "AI Prediction", "Real-time Alerts", "Settings"])


    # Fetch Data
    data = fetch_data()

    # ---------- Home Page ----------
    if page == "Home":
        st.subheader("📌 Latest Sensor Readings")

        if not data.empty:
            latest = data.iloc[-1]
            cols = st.columns(4)
            for i, field in enumerate(FIELDS):
                with cols[i % 4]:
                    st.metric(label=field, value=f"{latest[field]}")
            st.caption(f"Last updated: {latest['created_at']}")
        else:
            st.warning("No data available yet.")

    # ---------- Sensors Page ----------
    elif page == "Sensors":
        st.subheader("📈 Sensor Data Graphs")

        if not data.empty:
            # 🔽 CSV Download Button
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Sensor Data as CSV",
                data=csv,
                file_name='firework_sensor_data.csv',
                mime='text/csv',
                use_container_width=True
            )

            # Sensor Graph Dropdown
            sensor = st.sidebar.selectbox("Select Sensor", options=FIELDS)
            st.line_chart(data.set_index('created_at')[sensor])
        else:
            st.warning("No data available for export or graph.")

    # ---------- AI Prediction Page ----------
    elif page == "AI Prediction":
        st.subheader("🧠 AI-Based 1-Hour Prediction & Alerts")
        model = load_model()

        if model and not data.empty:
            latest = data.iloc[-1]

            # Real-time Prediction
            st.markdown("### 🔍 Real-time Prediction")
            input_data = latest[["Temperature", "Humidity", "Rain", "Flame", "Waterlevel", "Sound", "Voltage",
                                 "Current"]].values.reshape(1, -1)
            prediction = model.predict(input_data)[0]
            st.success(f"🤖 AI Prediction Output: **{prediction}**")

            # Fire Alert
            flame = latest["Flame"]
            if flame > 500:  # Example threshold
                st.error("🚨 FIRE ALERT DETECTED!")
            else:
                st.info("✅ No fire detected currently.")

            # Thunderstorm Alert
            if latest["Humidity"] > 70 and latest["Rain"] > 5 and latest["Sound"] > 500:
                st.warning("⚡ Thunderstorm possible in next 1 hour.")
            else:
                st.success("☀️ No thunderstorm expected in the next hour.")

            # Manual Input Section
            st.markdown("---")
            st.markdown("### ✍️ Manual Sensor Input for AI Prediction")
            col1, col2, col3, col4 = st.columns(4)
            temperature = col1.number_input("Temperature", value=30.0)
            humidity = col2.number_input("Humidity", value=50.0)
            rain = col3.number_input("Rain", value=0.0)
            flame = col4.number_input("Flame", value=200.0)
            water = col1.number_input("Waterlevel", value=30.0)
            sound = col2.number_input("Sound", value=100.0)
            voltage = col3.number_input("Voltage", value=220.0)
            current = col4.number_input("Current", value=0.5)

            if st.button("📡 Predict with Manual Input"):
                manual_data = np.array([[temperature, humidity, rain, flame, water, sound, voltage, current]])
                manual_pred = model.predict(manual_data)[0]
                st.success(f"📊 AI Prediction Output: **{manual_pred}**")

                if flame > 500:
                    st.error("🚨 FIRE ALERT DETECTED!")
                else:
                    st.info("✅ No fire based on input.")

                if humidity > 70 and rain > 5 and sound > 500:
                    st.warning("⚡ Thunderstorm may occur in next 1 hour.")
                else:
                    st.success("☀️ No thunderstorm expected.")

        else:
            st.warning("No model or data available for prediction.")

    # ---------- Real-time Alerts Page ----------
    elif page == "Real-time Alerts":
        st.subheader("🚨 Real-time Alerts via WhatsApp")

        # Set Alert Thresholds
        st.markdown("### 🔧 Set Alert Thresholds")
        temperature_threshold = st.number_input("Temperature Threshold", min_value=0.0, max_value=100.0, value=37.0)
        humidity_threshold = st.number_input("Humidity Threshold", min_value=0.0, max_value=100.0, value=80.0)
        flame_threshold = st.number_input("Flame Threshold", min_value=0.0, max_value=1.0, value=1.0)

        # Contact Information
        st.markdown("### 📞 Set Up WhatsApp Alert")
        to_whatsapp = st.text_input("Enter WhatsApp Number to Send Alerts", value="whatsapp:+916374930315")
        message = st.text_area("Enter Alert Message",
                               value="Alert! Sensor threshold exceeded. Please check the system.")

        if st.button("Send Test Alert"):
            send_whatsapp_alert(to_whatsapp, message)

        # Real-time Alert Logic
        if not data.empty:
            latest = data.iloc[-1]

            # Fire Alert
            if latest["Flame"] > flame_threshold:
                send_whatsapp_alert(to_whatsapp, f"🚨 FIRE ALERT DETECTED! Current Flame: {latest['Flame']}")

            # Temperature Alert
            if latest["Temperature"] > temperature_threshold:
                send_whatsapp_alert(to_whatsapp, f"⚠️ High Temperature Alert! Current Temp: {latest['Temperature']}°C")

            # Humidity Alert
            if latest["Humidity"] > humidity_threshold:
                send_whatsapp_alert(to_whatsapp, f"⚠️ High Humidity Alert! Current Humidity: {latest['Humidity']}%")

        else:
            st.warning("No data available for alert evaluation.")

    # ---------- Settings Page ----------
    elif page == "Settings":
        st.subheader("⚙️ Settings")

        st.markdown("Use this page to customize app preferences and configurations.")

        # Change login credentials (demo purpose - not persistent)
        st.markdown("### 🔐 Change Login Credentials")
        new_username = st.text_input("New Username", value="Lightning")
        new_password = st.text_input("New Password", value="Light@1234", type="password")

        if st.button("Update Credentials"):
            st.info("Note: In this demo, changes are not persisted.")
            st.success(f"Updated credentials to:\nUsername: {new_username}, Password: {new_password}")

        # Twilio settings
        st.markdown("### 📲 Twilio Settings")
        st.write("Update your Twilio SID and Auth Token for WhatsApp alerts.")

        sid_input = st.text_input("Twilio SID", value=SID)
        auth_token_input = st.text_input("Auth Token", value=AUTH_TOKEN, type="password")
        twilio_number = st.text_input("Twilio WhatsApp Number", value=TWILIO_PHONE_NUMBER)

        if st.button("Save Twilio Settings"):
            st.info("This change is for session only and not persistent in this version.")
            st.success("Twilio settings updated (session only).")

        # Logout
        st.markdown("### 🚪 Logout")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
