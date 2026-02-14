import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import os
from PIL import Image
from deepface import DeepFace
import tempfile
import sqlite3
import json
from streamlit_js_eval import get_geolocation
import requests

# -------------------------
# Database / Utility Setup
# -------------------------
DB_PATH = "attendance_system.db"

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            embedding TEXT NOT NULL,
            registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'Present',
            location TEXT,
            latitude REAL,
            longitude REAL,
            marked_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, date)
        )
    ''')
    
    conn.commit()
    conn.close()

@st.cache_resource
def load_models():
    """Initialize DeepFace (models are loaded automatically on first use)"""
    try:
        st.info("Loading face recognition models... (first time may take a few minutes)")
        init_database()
        return True
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return False

# -------------------------
# Face DB helpers
# -------------------------
def save_face_to_db(name, embedding):
    """Save face embedding to SQL database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        embedding_json = json.dumps(np.array(embedding).tolist())
        cursor.execute('''
            INSERT OR REPLACE INTO faces (name, embedding, registered_date)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (name, embedding_json))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving to database: {e}")
        return False
    finally:
        conn.close()

def load_faces_from_db():
    """Load all face embeddings from SQL database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT name, embedding FROM faces')
    rows = cursor.fetchall()
    conn.close()
    
    embeddings_dict = {}
    for name, embedding_json in rows:
        try:
            embeddings_dict[name] = np.array(json.loads(embedding_json))
        except Exception:
            embeddings_dict[name] = np.array(json.loads(embedding_json))
    
    return embeddings_dict

def delete_face_from_db(name):
    """Delete a face from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM faces WHERE name = ?', (name,))
    conn.commit()
    conn.close()

def clear_all_faces():
    """Clear all faces from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM faces')
    conn.commit()
    conn.close()

# -------------------------
# Image / embedding helpers
# -------------------------
def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def get_face_embedding(image):
    """Extract face embedding from image using DeepFace"""
    try:
        if isinstance(image, Image.Image):
            image = pil_to_cv2(image)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, image)
            tmp_path = tmp_file.name
        
        try:
            embedding_objs = DeepFace.represent(
                img_path=tmp_path,
                model_name='Facenet512',
                enforce_detection=True,
                detector_backend='retinaface'
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                embedding = embedding_objs[0]['embedding']
                return np.array(embedding)
            else:
                return None
                
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        st.error(f"Error extracting embedding: {e}")
        return None

def get_multiple_face_embeddings(image):
    """Extract embeddings for multiple faces in an image"""
    try:
        if isinstance(image, Image.Image):
            image = pil_to_cv2(image)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, image)
            tmp_path = tmp_file.name
        
        try:
            embedding_objs = DeepFace.represent(
                img_path=tmp_path,
                model_name='Facenet512',
                enforce_detection=True,
                detector_backend='retinaface'
            )
            
            embeddings = []
            face_areas = []
            
            for obj in embedding_objs:
                embeddings.append(np.array(obj['embedding']))
                face_areas.append(obj.get('facial_area', {}))
            
            return embeddings, face_areas
                
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        st.error(f"Error extracting embeddings: {e}")
        return [], []

# -------------------------
# Matching helpers
# -------------------------
def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    embedding1 = np.array(embedding1).flatten()
    embedding2 = np.array(embedding2).flatten()
    
    if embedding1.shape != embedding2.shape:
        st.warning(f"Embedding dimension mismatch: {embedding1.shape} vs {embedding2.shape}. Please clear and re-register faces.")
        return 0
    
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

def find_matching_face(query_embedding, embeddings_dict, threshold=0.70):
    """Find matching face in database using cosine similarity"""
    best_match = None
    best_similarity = -1
    
    for name, stored_embedding in embeddings_dict.items():
        similarity = cosine_similarity(query_embedding, stored_embedding)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = name
    
    if best_similarity >= threshold:
        return best_match, best_similarity
    else:
        return None, best_similarity

# -------------------------
# Attendance DB helpers
# -------------------------
def mark_attendance(name, selected_date, location=None, latitude=None, longitude=None):
    """Mark attendance in SQL database for a specific date with location"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.now()
    date_str = selected_date.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    
    try:
        
        cursor.execute('''
            SELECT id FROM attendance WHERE name = ? AND date = ?
        ''', (name, date_str))
        
        if cursor.fetchone():
            conn.close()
            return False, "Already marked attendance for this date"
        
        
        cursor.execute('''
            INSERT INTO attendance (name, date, time, status, location, latitude, longitude)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, date_str, time_str, 'Present', location, latitude, longitude))
        
        conn.commit()
        conn.close()
        return True, "Attendance marked successfully"
        
    except Exception as e:
        conn.close()
        return False, f"Error: {e}"

def delete_attendance(name, selected_date):
    """Delete attendance record for a specific person and date"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    date_str = selected_date.strftime('%Y-%m-%d')
    
    cursor.execute('''
        DELETE FROM attendance WHERE name = ? AND date = ?
    ''', (name, date_str))
    
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    if deleted:
        return True, f"Attendance deleted for {name}"
    else:
        return False, f"No attendance record found for {name}"

def update_attendance_time(name, selected_date, new_time):
    """Update the time for an attendance record"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    date_str = selected_date.strftime('%Y-%m-%d')
    
    cursor.execute('''
        UPDATE attendance SET time = ? WHERE name = ? AND date = ?
    ''', (new_time, name, date_str))
    
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    if updated:
        return True, f"Time updated for {name}"
    else:
        return False, f"No attendance record found for {name}"

def update_attendance_location(name, selected_date, location, latitude=None, longitude=None):
    """Update location for an attendance record"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    date_str = selected_date.strftime('%Y-%m-%d')
    
    cursor.execute('''
        UPDATE attendance SET location = ?, latitude = ?, longitude = ?
        WHERE name = ? AND date = ?
    ''', (location, latitude, longitude, name, date_str))
    
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    if updated:
        return True, f"Location updated for {name}"
    else:
        return False, f"No attendance record found for {name}"

def get_attendance_for_date(selected_date):
    """Get all attendance records for a specific date"""
    conn = sqlite3.connect(DB_PATH)
    
    date_str = selected_date.strftime('%Y-%m-%d')
    
    df = pd.read_sql_query('''
        SELECT name, date, time, status, location, latitude, longitude
        FROM attendance
        WHERE date = ?
        ORDER BY time
    ''', conn, params=(date_str,))
    
    conn.close()
    return df

def clear_attendance_for_date(selected_date):
    """Clear all attendance records for a specific date"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    date_str = selected_date.strftime('%Y-%m-%d')
    
    cursor.execute('DELETE FROM attendance WHERE date = ?', (date_str,))
    conn.commit()
    conn.close()

# -------------------------
# Reverse Geocoding (Nominatim)
# -------------------------
def reverse_geocode(latitude, longitude, max_length=200):
    """
    Reverse-geocode lat/lon into a human-readable address using OpenStreetMap Nominatim.
    Returns a short address string or None on failure.
    """
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "format": "jsonv2",
            "lat": float(latitude),
            "lon": float(longitude),
            "zoom": 18,
            "addressdetails": 0
        }
        headers = {
            "User-Agent": "FaceAttendanceApp/1.0 (contact: you@example.com)"
        }
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            display_name = data.get("display_name", "")
            if display_name:
                # shorten if excessively long
                return (display_name[:max_length]).strip()
        return None
    except Exception:
        return None

# -------------------------
# Main App
# -------------------------
def main():
    st.set_page_config(page_title="Face Recognition Attendance System", layout="wide")
    
    st.title("🎯 Face Recognition Attendance System")
    st.markdown("---")
    
    models_loaded = load_models()
    
    if not models_loaded:
        st.error("Failed to load face recognition models. Please check your installation.")
        return
    
    embeddings_dict = load_faces_from_db()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page",
                                ["Register Face", "Mark Attendance", "View Attendance", "Manage Database"])
    
    if page == "Register Face":
        st.header("👤 Register New Face")
        
        name = st.text_input("Enter Name:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("Register Face from Upload"):
                    if name:
                        with st.spinner("Processing face..."):
                            embedding = get_face_embedding(image)
                            
                            if embedding is not None:
                                if save_face_to_db(name, embedding):
                                    st.success(f"Face registered successfully for {name}!")
                                else:
                                    st.error("Failed to save face to database.")
                            else:
                                st.error("No face detected in the image. Please upload a clear face image.")
                    else:
                        st.error("Please enter a name.")
        
        with col2:
            st.subheader("Camera Capture")
            camera_input = st.camera_input("Take a picture")
            
            if camera_input is not None:
                image = Image.open(camera_input).convert("RGB")
                
                if st.button("Register from Camera"):
                    if name:
                        with st.spinner("Processing face..."):
                            embedding = get_face_embedding(image)
                            
                            if embedding is not None:
                                if save_face_to_db(name, embedding):
                                    st.success(f"Face registered successfully for {name}!")
                                else:
                                    st.error("Failed to save face to database.")
                            else:
                                st.error("No face detected in the image. Please take a clear face photo.")
                    else:
                        st.error("Please enter a name.")
    
    elif page == "Mark Attendance":
        st.header("✅ Mark Attendance")
        
        if not embeddings_dict:
            st.warning("No faces registered yet. Please register faces first.")
            return
        
        
        st.sidebar.subheader("⚙️ Recognition Settings")
        threshold = st.sidebar.slider(
            "Recognition Threshold",
            min_value=0.50,
            max_value=0.90,
            value=0.70,
            step=0.05,
            help="Higher threshold = stricter matching (fewer false positives)"
        )
        
        st.sidebar.info(f"Current threshold: **{threshold:.0%}**")
        
        # -------------------------
        # Geolocation capture & reverse geocode
        # -------------------------
        st.subheader("📍 Device Location")
        loc = get_geolocation()  
        latitude = None
        longitude = None
        accuracy = None
        location_name = None
        
        if loc and isinstance(loc, dict) and "coords" in loc:
            coords = loc["coords"]
            latitude = coords.get("latitude")
            longitude = coords.get("longitude")
            accuracy = coords.get("accuracy")
            
            
            if latitude is not None and longitude is not None:
                human_addr = reverse_geocode(latitude, longitude)
                if human_addr:
                    location_name = human_addr
                else:
                    
                    location_name = f"{latitude:.6f}, {longitude:.6f}"
            
            st.success(
                f"📍 Location Captured — Latitude: {latitude:.6f}, Longitude: {longitude:.6f} (±{accuracy:.0f} m)\n\n"
                f"**Address:** {location_name}"
            )
        else:
            st.warning("⏳ Waiting for device location... Please allow location access in your browser.")
            
            if st.button("Retry getting location"):
                st.experimental_rerun()
        
        
        st.subheader("📅 Select Date for Attendance")
        
        col_date1, col_date2, col_date3 = st.columns([1, 1, 1])
        
        with col_date1:
            if st.button("📍 Today", use_container_width=True, key="today_btn"):
                st.session_state.selected_date = date.today()
        
        with col_date2:
            if st.button("➡️ Tomorrow", use_container_width=True, key="tomorrow_btn"):
                st.session_state.selected_date = date.today() + timedelta(days=1)
        
        with col_date3:
            st.markdown("""
                <div style='padding: 10px; border: 1px solid #4a5568; border-radius: 5px; text-align: center;'>
                    <p style='margin: 0; font-weight: bold;'>📆 Calendar View</p>
                </div>
            """, unsafe_allow_html=True)
            calendar_date = st.date_input(
                "Select date",
                value=st.session_state.get('selected_date', date.today()),
                key="calendar_picker",
                label_visibility="collapsed"
            )
            st.session_state.selected_date = calendar_date
        
        selected_date = st.session_state.get('selected_date', date.today())
        st.info(f"📅 Marking attendance for: **{selected_date.strftime('%A, %B %d, %Y')}**")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        # -------------------------
        # Upload image flow
        # -------------------------
        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader("Upload image for attendance", type=['jpg', 'jpeg', 'png'], key="upload_attend")
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Image for Attendance", use_container_width=True)
                
                if st.button("Check Attendance from Upload"):
                    
                    if latitude is None or longitude is None:
                        st.error("❌ Cannot mark attendance — GPS location not captured.")
                        st.warning("📍 Please allow location access and wait for GPS to load above.")
                    else:
                        with st.spinner("Recognizing faces..."):
                            embeddings, face_areas = get_multiple_face_embeddings(image)
                            
                            if embeddings:
                                st.success(f"✅ Detected {len(embeddings)} face(s) in the image")
                                
                                recognized_people = []
                                attendance_results = []
                                unknown_faces = []
                                
                                for i, embedding in enumerate(embeddings):
                                    matched_name, similarity = find_matching_face(embedding, embeddings_dict, threshold=threshold)
                                    
                                    if matched_name:
                                        recognized_people.append({
                                            'name': matched_name,
                                            'confidence': similarity,
                                            'face_number': i + 1
                                        })
                                        
                                        success, message = mark_attendance(
                                            matched_name, 
                                            selected_date, 
                                            location_name if location_name else None,
                                            latitude,
                                            longitude
                                        )
                                        attendance_results.append({
                                            'name': matched_name,
                                            'success': success,
                                            'message': message
                                        })
                                    else:
                                        unknown_faces.append({
                                            'face_number': i + 1,
                                            'confidence': similarity
                                        })
                                
                                if recognized_people:
                                    st.subheader("👥 Recognized Faces:")
                                    for person in recognized_people:
                                        st.success(f"✅ Face {person['face_number']}: **{person['name']}** (Match: {person['confidence']:.1%})")
                                    
                                    st.subheader("📝 Attendance Results:")
                                    for result in attendance_results:
                                        if result['success']:
                                            st.success(f"✅ {result['name']}: {result['message']}")
                                        else:
                                            st.warning(f"⚠️ {result['name']}: {result['message']}")
                                
                                if unknown_faces:
                                    st.subheader("❌ Unknown/Unregistered Faces:")
                                    for unknown in unknown_faces:
                                        st.error(f"❌ Face {unknown['face_number']}: **Not Registered** (Best match: {unknown['confidence']:.1%})")
                                    st.warning("⚠️ **No attendance marked for unknown faces.** Please register them first.")
                            else:
                                st.error("❌ No faces detected in the image. Please upload an image with visible faces.")
        
        # -------------------------
        # Camera flow
        # -------------------------
        with col2:
            st.subheader("Camera Capture")
            camera_input = st.camera_input("Take picture for attendance", key="camera_attend")
            
            if camera_input is not None:
                image = Image.open(camera_input).convert("RGB")
                
                if st.button("Mark Attendance from Camera"):
                    
                    if latitude is None or longitude is None:
                        st.error("❌ Cannot mark attendance — GPS location not captured.")
                        st.warning("📍 Please allow location access and wait for GPS to load above.")
                        st.stop()
                    
                    with st.spinner("Recognizing faces..."):
                        embeddings, face_areas = get_multiple_face_embeddings(image)
                        
                        if embeddings:
                            st.success(f"✅ Detected {len(embeddings)} face(s) in the image")
                            
                            recognized_people = []
                            attendance_results = []
                            unknown_faces = []
                            
                            for i, embedding in enumerate(embeddings):
                                matched_name, similarity = find_matching_face(embedding, embeddings_dict, threshold=threshold)
                                
                                if matched_name:
                                    recognized_people.append({
                                        'name': matched_name,
                                        'confidence': similarity,
                                        'face_number': i + 1
                                    })
                                    
                                    success, message = mark_attendance(
                                        matched_name, 
                                        selected_date,
                                        location_name if location_name else None,
                                        latitude,
                                        longitude
                                    )
                                    attendance_results.append({
                                        'name': matched_name,
                                        'success': success,
                                        'message': message
                                    })
                                else:
                                    unknown_faces.append({
                                        'face_number': i + 1,
                                        'confidence': similarity
                                    })
                            
                            if recognized_people:
                                st.subheader("👥 Recognized Faces:")
                                for person in recognized_people:
                                    st.success(f"✅ Face {person['face_number']}: **{person['name']}** (Match: {person['confidence']:.1%})")
                                
                                st.subheader("📝 Attendance Results:")
                                for result in attendance_results:
                                    if result['success']:
                                        st.success(f"✅ {result['name']}: {result['message']}")
                                    else:
                                        st.warning(f"⚠️ {result['name']}: {result['message']}")
                            
                            if unknown_faces:
                                st.subheader("❌ Unknown/Unregistered Faces:")
                                for unknown in unknown_faces:
                                    st.error(f"❌ Face {unknown['face_number']}: **Not Registered** (Best match: {unknown['confidence']:.1%})")
                                st.warning("⚠️ **No attendance marked for unknown faces.** Please register them first.")
                        else:
                            st.error("❌ No faces detected in the image. Please take a photo with visible faces.")
    
    elif page == "View Attendance":
        st.header("📊 View Attendance Records")
        
        today = date.today()
        selected_date = st.date_input("Select Date", value=today, key="view_date")
        
        df = get_attendance_for_date(selected_date)
        
        if not df.empty:
            st.subheader(f"Attendance for {selected_date.strftime('%A, %B %d, %Y')}")
            
            for idx, row in df.iterrows():
                col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 2, 1, 1])
                
                with col1:
                    st.write(f"**{row['name']}**")
                
                with col2:
                    st.write(f"📅 {row['date']}")
                
                with col3:
                    new_time = st.text_input(
                        "Time",
                        value=row['time'],
                        key=f"time_{idx}",
                        label_visibility="collapsed"
                    )
                    if new_time != row['time']:
                        if st.button("💾", key=f"save_time_{idx}"):
                            success, message = update_attendance_time(row['name'], selected_date, new_time)
                            if success:
                                st.success(message)
                                st.rerun()
                
                with col4:
                    location_display = row['location'] if pd.notna(row['location']) and row['location'] else "Not recorded"
                    if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                        coords_text = f" ({row['latitude']:.4f}, {row['longitude']:.4f})"
                        location_display += coords_text
                    st.write(f"📍 {location_display}")
                    
                    
                    if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                        maps_url = f"https://www.google.com/maps?q={row['latitude']},{row['longitude']}"
                        st.markdown(f"[🗺️ Open in Google Maps]({maps_url})")
                
                with col5:
                    st.write(f"✅ {row['status']}")
                
                with col6:
                    if st.button("🗑️", key=f"delete_{idx}"):
                        success, message = delete_attendance(row['name'], selected_date)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                
                st.markdown("---")
            
            st.subheader("Summary")
            col_sum1, col_sum2 = st.columns(2)
            with col_sum1:
                st.metric("Total Present", len(df))
            with col_sum2:
                locations = df['location'].dropna().unique()
                st.metric("Unique Locations", len(locations))
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"attendance_{selected_date.strftime('%Y_%m_%d')}.csv",
                mime='text/csv'
            )
            
            if st.button("🗑️ Clear All Attendance for This Date", type="secondary"):
                clear_attendance_for_date(selected_date)
                st.success(f"All attendance records for {selected_date} have been deleted!")
                st.rerun()
        else:
            st.info(f"No attendance records found for {selected_date.strftime('%A, %B %d, %Y')}")
    
    elif page == "Manage Database":
        st.header("🗄️ Manage Face Database")
        
        st.subheader("Registered Faces")
        if embeddings_dict:
            for i, name in enumerate(embeddings_dict.keys()):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{i + 1}. {name}")
                with col2:
                    if st.button(f"🗑️ Delete", key=f"delete_{name}"):
                        delete_face_from_db(name)
                        st.success(f"Deleted {name}")
                        st.rerun()
            
            st.subheader("Database Statistics")
            st.metric("Total Registered Faces", len(embeddings_dict))
            
            if st.button("🗑️ Clear All Faces", type="secondary"):
                clear_all_faces()
                st.success("All faces cleared from database!")
                st.rerun()
        else:
            st.info("No faces registered yet.")
        
        st.markdown("---")
        st.subheader("Database Information")
        st.info("📊 Using SQLite database: `attendance_system.db`\n\n"
                "• **faces** table: Stores face embeddings\n"
                "• **attendance** table: Stores attendance records with location data")

if __name__ == "__main__":
    main()
