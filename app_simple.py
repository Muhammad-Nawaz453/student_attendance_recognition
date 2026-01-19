from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import cv2
import numpy as np
import os
from datetime import datetime
import sqlite3
import base64
from functools import wraps
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
DATASET_FOLDER = 'dataset'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer_trained = False

def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  student_id TEXT UNIQUE NOT NULL,
                  name TEXT NOT NULL,
                  email TEXT,
                  class TEXT,
                  image_path TEXT,
                  face_id INTEGER,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  student_id TEXT NOT NULL,
                  date DATE NOT NULL,
                  time TIME NOT NULL,
                  status TEXT DEFAULT 'Present',
                  FOREIGN KEY (student_id) REFERENCES students(student_id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS admins
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    
    c.execute("SELECT * FROM admins WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO admins (username, password) VALUES ('admin', 'admin123')")
    
    conn.commit()
    conn.close()

init_db()

def train_recognizer():
    """Train the face recognizer with all registered students"""
    global recognizer, recognizer_trained
    
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT face_id, image_path FROM students WHERE image_path IS NOT NULL")
    students = c.fetchall()
    conn.close()
    
    if not students:
        return False
    
    faces = []
    labels = []
    
    for face_id, image_path in students:
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_regions = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in face_regions:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                faces.append(face)
                labels.append(face_id)
    
    if faces:
        recognizer.train(faces, np.array(labels))
        recognizer_trained = True
        return True
    return False

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        c.execute("SELECT * FROM admins WHERE username=? AND password=?", (username, password))
        admin = c.fetchone()
        conn.close()
        
        if admin:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM students")
    total_students = c.fetchone()[0]
    
    c.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date=?", (datetime.now().date(),))
    present_today = c.fetchone()[0]
    
    conn.close()
    
    return render_template('dashboard.html', 
                         total_students=total_students,
                         present_today=present_today)

@app.route('/register_student', methods=['GET', 'POST'])
@login_required
def register_student():
    if request.method == 'POST':
        student_id = request.form['student_id']
        name = request.form['name']
        email = request.form['email']
        class_name = request.form['class']
        
        if 'image' in request.files:
            image = request.files['image']
            if image.filename:
                image_path = os.path.join(DATASET_FOLDER, f"{student_id}.jpg")
                image.save(image_path)
                
                # Verify face in image
                img = cv2.imread(image_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) == 0:
                    os.remove(image_path)
                    return render_template('register_student.html', 
                                         error='No face detected in image. Please upload a clear face photo.')
                
                # Save to database
                try:
                    conn = sqlite3.connect('attendance.db')
                    c = conn.cursor()
                    
                    # Get next face_id
                    c.execute("SELECT MAX(face_id) FROM students")
                    max_id = c.fetchone()[0]
                    face_id = (max_id + 1) if max_id else 1
                    
                    c.execute("""INSERT INTO students (student_id, name, email, class, image_path, face_id)
                                VALUES (?, ?, ?, ?, ?, ?)""",
                             (student_id, name, email, class_name, image_path, face_id))
                    conn.commit()
                    conn.close()
                    
                    # Retrain recognizer
                    train_recognizer()
                    
                    return render_template('register_student.html', 
                                         success='Student registered successfully!')
                except sqlite3.IntegrityError:
                    return render_template('register_student.html', 
                                         error='Student ID already exists!')
    
    return render_template('register_student.html')

@app.route('/take_attendance')
@login_required
def take_attendance():
    return render_template('take_attendance.html')

@app.route('/recognize_face', methods=['POST'])
@login_required
def recognize_face():
    global recognizer_trained
    
    if not recognizer_trained:
        train_recognizer()
    
    if not recognizer_trained:
        return jsonify({'success': False, 'message': 'No students registered yet'})
    
    data = request.json
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return jsonify({'success': False, 'message': 'No face detected'})
    
    # Get the first face
    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (200, 200))
    
    # Recognize face
    label, confidence = recognizer.predict(face)
    
    # Lower confidence is better (0-100 scale, typically < 50 is good match)
    if confidence < 80:
        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        c.execute("SELECT student_id, name FROM students WHERE face_id=?", (int(label),))
        student = c.fetchone()
        
        if student:
            student_id, name = student
            
            # Check if already marked present today
            today = datetime.now().date()
            c.execute("SELECT * FROM attendance WHERE student_id=? AND date=?", 
                     (student_id, today))
            
            if c.fetchone():
                conn.close()
                return jsonify({'success': True, 
                              'message': f'{name} already marked present today',
                              'student_id': student_id,
                              'name': name,
                              'confidence': round(100 - confidence, 2)})
            
            # Mark attendance
            current_time = datetime.now().time()
            c.execute("""INSERT INTO attendance (student_id, date, time, status)
                        VALUES (?, ?, ?, 'Present')""",
                     (student_id, today, current_time))
            conn.commit()
            conn.close()
            
            return jsonify({'success': True, 
                          'message': f'Attendance marked for {name}',
                          'student_id': student_id,
                          'name': name,
                          'confidence': round(100 - confidence, 2)})
    
    return jsonify({'success': False, 'message': 'Face not recognized'})

@app.route('/view_attendance')
@login_required
def view_attendance():
    # Get date from query parameter or use today's date
    date_str = request.args.get('date')
    if date_str:
        date = date_str
    else:
        date = str(datetime.now().date())
    
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("""SELECT a.student_id, s.name, s.class, a.time, a.status
                 FROM attendance a
                 JOIN students s ON a.student_id = s.student_id
                 WHERE a.date = ?
                 ORDER BY a.time DESC""", (date,))
    records = c.fetchall()
    conn.close()
    
    return render_template('view_attendance.html', records=records, date=date)

@app.route('/students')
@login_required
def students():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT student_id, name, email, class FROM students")
    students = c.fetchall()
    conn.close()
    
    return render_template('students.html', students=students)

@app.route('/debug_attendance')
@login_required
def debug_attendance():
    """Debug route to check all attendance records"""
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM attendance")
    all_records = c.fetchall()
    conn.close()
    
    return jsonify({
        'total_records': len(all_records),
        'records': all_records
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)