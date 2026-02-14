import streamlit as st
import streamlit.components.v1 as components

st.title("📍 Test Location — Direct Component Method")

# Load streamlit-geolocation component manually
geolocation_component = components.declare_component(
    "geolocation",
    path="venv/lib/python3.11/site-packages/streamlit_geolocation/frontend/build"
)

result = geolocation_component()

st.write("### Raw result:")
st.write(result)

if result and "coords" in result and result["coords"]:
    coords = result["coords"]
    st.success(f"Latitude: {coords['latitude']}, Longitude: {coords['longitude']}")
else:
    st.warning("No location yet — click the button and allow GPS access.")

