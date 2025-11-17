from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
import mysql.connector
from datetime import datetime, timedelta, date, time
import os
import uuid
import hashlib
import base64
import cv2
import numpy as np
import traceback
import urllib.parse 
import re
import config

# ✨ [NEW] Thư viện AI cho Liveness Detection 2.0
import mediapipe as mp

# Khởi tạo thư viện nhận diện khuôn mặt
try:
    import face_recognition
    FACE_LIB = 'face_recognition'
except Exception as e:
    print('face_recognition không khả dụng, sử dụng opencv LBPH:', e)
    FACE_LIB = 'opencv_fallback'

app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key-change-this-in-production'

# ===================================================================
# ✨ CẤU HÌNH MEDIAPIPE (LIVENESS 2.0)
# ===================================================================
# Khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,       # Xử lý từng ảnh tĩnh (do nhận từ request)
    max_num_faces=1,              # Chỉ xử lý 1 khuôn mặt để tối ưu
    refine_landmarks=True,        # Lấy điểm chi tiết mắt (iris) để tính toán chính xác
    min_detection_confidence=0.5
)

# Các điểm landmark cho mắt (MediaPipe indexes)
# Mắt trái
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Mắt phải
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, indices, img_w, img_h):
    """
    Tính tỷ lệ mở mắt (Eye Aspect Ratio - EAR)
    EAR = (khoảng cách dọc 1 + khoảng cách dọc 2) / (2 * khoảng cách ngang)
    """
    try:
        # Lấy tọa độ các điểm
        coords = []
        for idx in indices:
            lm = landmarks[idx]
            # Chuyển đổi tọa độ chuẩn hóa (0-1) sang tọa độ pixel
            coords.append(np.array([lm.x * img_w, lm.y * img_h]))

        # Tính khoảng cách dọc (vertical)
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])

        # Tính khoảng cách ngang (horizontal)
        h = np.linalg.norm(coords[0] - coords[3])

        # Tính EAR
        if h == 0: return 0.0
        ear = (v1 + v2) / (2.0 * h)
        return ear
    except Exception as e:
        print(f"Lỗi tính EAR: {e}")
        return 0.0

def check_head_pose(landmarks, img_w, img_h):
    """
    Kiểm tra xem người dùng có đang nhìn thẳng không (Head Pose Estimation đơn giản)
    Dựa vào độ lệch của mũi so với trung tâm hai mắt.
    """
    try:
        # Lấy các điểm quan trọng: Mũi (1), Mắt trái (33), Mắt phải (263)
        nose_tip = landmarks[1]
        left_eye_outer = landmarks[33]
        right_eye_outer = landmarks[263]

        # Tính trung tâm X của hai mắt
        eye_center_x = (left_eye_outer.x + right_eye_outer.x) / 2
        
        # Độ lệch của mũi so với trung tâm mắt
        nose_offset_x = nose_tip.x - eye_center_x
        
        # Ngưỡng chấp nhận (Look Straight). Nếu lệch quá > 0.08 là đang quay đầu
        # Giá trị 0.08 là ngưỡng thực nghiệm, có thể điều chỉnh
        is_looking_straight = abs(nose_offset_x) < 0.08
        
        return is_looking_straight
    except Exception as e:
        print(f"Lỗi kiểm tra hướng đầu: {e}")
        return False

# ===================================================================
# ✨ LOGIC GIÁ VÉ VÀ MA TRẬN GIÁ
# ===================================================================
# Ánh xạ tên ga trong CSDL với bảng giá
STATIONS_LIST = [
    'Ga Bến Thành', 'Ga Nhà hát Thành phố', 'Ga Ba Son', 'Ga Công viên Văn Thánh',
    'Ga Tân Cảng', 'Ga Thảo Điền', 'Ga An Phú', 'Ga Rạch Chiếc', 'Ga Phước Long',
    'Ga Bình Thái', 'Ga Thủ Đức', 'Ga Khu Công nghệ cao', 'Ga Đại học Quốc gia', 'Ga Bến xe Suối Tiên'
]

# Ma trận giá (x 1000 VNĐ)
PRICE_MATRIX = [
    # Cột 1 -> 14 (Bến Thành -> Suối Tiên)
    [7, 7, 7, 7, 7, 7, 7, 9, 10, 12, 14, 16, 18, 20], # Ga Bến Thành
    [7, 7, 7, 7, 7, 7, 7, 8, 10, 11, 13, 15, 17, 20], # Ga Nhà hát TP
    [7, 7, 7, 7, 7, 7, 7, 7, 9, 10, 12, 15, 16, 18], # Ga Ba Son
    [7, 7, 7, 7, 7, 7, 7, 7, 8, 10, 13, 14, 17, 18], # Ga Văn Thánh
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 9, 12, 13, 16, 17], # Ga Tân Cảng
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 10, 12, 14, 16], # Ga Thảo Điền
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9, 11, 13, 14], # Ga An Phú
    [9, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9, 11, 13], # Ga Rạch Chiếc
    [10, 10, 9, 7, 7, 7, 7, 7, 7, 7, 7, 8, 10, 11], # Ga Phước Long
    [12, 11, 10, 8, 9, 8, 7, 7, 7, 7, 7, 7, 8, 10], # Ga Bình Thái
    [14, 13, 12, 10, 12, 10, 9, 7, 7, 7, 7, 7, 7, 8], # Ga Thủ Đức
    [16, 15, 15, 13, 13, 12, 11, 9, 8, 7, 7, 7, 7, 7], # Ga CNC
    [18, 17, 16, 14, 16, 14, 13, 11, 9, 8, 7, 7, 7, 7], # Ga ĐH Quốc gia
    [20, 20, 18, 17, 17, 16, 14, 13, 11, 10, 8, 7, 7, 7]  # Ga Suối Tiên
]

def get_ticket_price(from_station, to_station):
    """Tính giá vé lượt dựa trên ma trận."""
    try:
        idx_from = STATIONS_LIST.index(from_station)
        idx_to = STATIONS_LIST.index(to_station)
        price_in_thousand = PRICE_MATRIX[idx_from][idx_to]
        # Giảm 1.000đ theo chú thích (*)
        return (price_in_thousand * 1000) - 1000 
    except (ValueError, IndexError):
        # Trả về -1 nếu không tìm thấy ga (để báo lỗi)
        return -1 

# ===================================================================
# ✨ GIAI ĐOẠN 3: LOGIC "AI" MÔ PHỎNG LỊCH TÀU
# ===================================================================
def get_train_frequency(selected_datetime):
    """
    Phân tích ngày và giờ để đưa ra gợi ý tần suất tàu.
    Dựa trên lịch tàu do người dùng cung cấp.
    """
    try:
        day_of_week = selected_datetime.weekday() # Thứ 2 = 0, Chủ Nhật = 6
        time_of_day = selected_datetime.time()
        
        is_weekend = (day_of_week >= 5) # Thứ 7 hoặc Chủ Nhật
        
        # --- Lịch Tàu Ngày Thường (Thứ 2 - Thứ 6) ---
        if not is_weekend:
            # Giờ cao điểm (5 phút/chuyến)
            if (time(7, 0) <= time_of_day <= time(9, 0)) or (time(16, 30) <= time_of_day <= time(18, 30)):
                return 5, "Giờ cao điểm (5 phút/chuyến). Ga có thể đông, bạn nên đến sớm."
            # Giờ bình thường (10 phút/chuyến)
            elif (time(5, 0) <= time_of_day < time(7, 0)) or (time(9, 0) < time_of_day < time(16, 30)) or (time(18, 30) < time_of_day <= time(22, 0)):
                return 10, "Giờ bình thường (10 phút/chuyến)."
            # Giờ thấp điểm (15 phút/chuyến)
            else:
                return 15, "Giờ thấp điểm (15 phút/chuyến)."
        
        # --- Lịch Tàu Cuối Tuần & Lễ (Thứ 7, CN) ---
        else:
            # Giờ cao điểm (8 phút/chuyến)
            if (time(7, 0) <= time_of_day <= time(9, 0)) or (time(17, 0) <= time_of_day <= time(19, 0)):
                return 8, "Giờ cao điểm cuối tuần (8 phút/chuyến)."
            # Giờ bình thường (10 phút/chuyến)
            elif (time(9, 0) < time_of_day < time(17, 0)):
                return 10, "Giờ bình thường cuối tuần (10 phút/chuyến)."
            # Giờ thấp điểm (15 phút/chuyến)
            else:
                return 15, "Giờ thấp điểm cuối tuần (15 phút/chuyến)."
                
    except Exception as e:
        print(f"Lỗi get_train_frequency: {e}")
        return None, "Không thể phân tích giờ tàu."

# ===================================================================
# ✨ BIẾN TOÀN CỤC
# ===================================================================
known_face_data = [] # Sẽ lưu danh sách (username, encoding) để nhận diện

# ===================================================================
# ✨ HÀM HỖ TRỢ CƠ SỞ DỮ LIỆU
# ===================================================================

def get_db():
    """Kết nối đến database MySQL."""
    return mysql.connector.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        database=config.DB_NAME,
        autocommit=True 
    )

def hash_pw(pw):
    """Mã hóa mật khẩu bằng SHA256."""
    return hashlib.sha256(pw.encode()).hexdigest()

def load_known_faces():
    """
    Tải trước tất cả khuôn mặt đã đăng ký vào RAM để nhận diện nhanh hơn.
    """
    global known_face_data
    known_face_data = []
   
    if not os.path.exists(config.FACES_DIR):
        return

    print("Đang tải dữ liệu khuôn mặt...")
    for fname in os.listdir(config.FACES_DIR):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
       
        try:
            username = os.path.splitext(fname)[0]
            image_path = os.path.join(config.FACES_DIR, fname)
            known_image = cv2.imread(image_path)
           
            if known_image is None:
                print(f"Không thể đọc file: {fname}")
                continue
           
            rgb_image = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)
           
            if encodings:
                known_face_data.append((username, encodings[0]))
                print(f"Đã tải khuôn mặt: {username}")
            else:
                print(f"Không tìm thấy khuôn mặt trong file: {fname}")
        except Exception as e:
            print(f"Lỗi khi tải {fname}: {e}")
    print(f"Hoàn tất tải {len(known_face_data)} khuôn mặt.")


def detect_station_columns():
    """Phát hiện tên cột của bảng `stations`."""
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=%s AND TABLE_NAME='stations'", (config.DB_NAME,))
        cols = [r[0] for r in cur.fetchall()]
        cur.close()
        db.close()
        if 'station_name' in cols:
            id_col = 'station_id' if 'station_id' in cols else 'id'
            return (id_col, 'station_name')
        if 'id' in cols and 'name' in cols:
            return ('id', 'name')
        if len(cols) >= 2:
            return (cols[0], cols[1])
    except Exception as e:
        print('detect_station_columns error:', e)
    return ('station_id', 'station_name')


def get_table_columns(table_name):
    """Lấy danh sách tên cột của một bảng."""
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s", (config.DB_NAME, table_name))
        cols = set([r[0] for r in cur.fetchall()])
        cur.close()
        db.close()
        return cols
    except Exception as e:
        print(f'get_table_columns({table_name}) error:', e)
        return set()


def get_all_stations():
    """Lấy danh sách các ga."""
    id_col, name_col = ('station_id', 'station_name') 
    db = get_db()
    cur = db.cursor(dictionary=True)
    try:
        q = f"SELECT {id_col} AS station_id, {name_col} AS station_name FROM stations ORDER BY {id_col}"
        cur.execute(q)
        stations = cur.fetchall()
        return stations
    except Exception as e:
        print('get_all_stations error:', e)
        return []
    finally:
        cur.close()
        db.close()


def ensure_stations(station_names):
    """Đảm bảo các ga mặc định tồn tại (Đã cập nhật theo CSDL mới)."""
    id_col, name_col = ('station_id', 'station_name')
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute(f"SELECT {name_col} FROM stations")
        existing = set([row[0] for row in cur.fetchall()])
        for s in station_names:
            if s not in existing:
                try:
                    cur.execute(f"INSERT INTO stations ({name_col}) VALUES (%s)", (s,))
                    print(f"Đã thêm ga mới: {s}")
                except Exception as e:
                    print(f'Unable to insert station {s}:', e)
        db.commit()
    except Exception as e:
        print('ensure_stations error:', e)
    finally:
        cur.close()
        db.close()


def ensure_face_data_table():
    """Đảm bảo bảng face_data tồn tại."""
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS face_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL UNIQUE,
                face_encoding LONGBLOB,
                photo_path VARCHAR(255),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        ''')
        cols = get_table_columns('users')
        if 'id' in cols:
            try:
                cur.execute('ALTER TABLE face_data ADD CONSTRAINT fk_face_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE')
            except Exception:
                pass
        db.commit()
    except Exception as e:
        print('ensure_face_data_table error:', e)
    finally:
        try: cur.close(); db.close()
        except Exception: pass


def ensure_tickets_schema():
    """Đảm bảo bảng tickets có các cột cần thiết (Đã cập nhật)."""
    try:
        cols = get_table_columns('tickets')
        db = get_db()
        cur = db.cursor()
        alters = []
        want = {
            'user_id': 'INT', 'ticket_type': "ENUM('monthly','single')", 'purchase_time': 'DATETIME',
            'valid_from': 'DATE', 'from_station_name': 'VARCHAR(255)',
            'to_station_name': 'VARCHAR(255)', 'status': 'VARCHAR(20)',
            'purchase_price': 'INT', 'used': 'TINYINT(1)', 'trip_code': 'VARCHAR(100)',
            'expected_departure_time': 'DATETIME'
        }
        for k, ddl in want.items():
            if k not in cols:
                alters.append(f"ADD COLUMN {k} {ddl}")
        if alters:
            sql = 'ALTER TABLE tickets ' + ', '.join(alters)
            try:
                cur.execute(sql)
                db.commit()
                print("Đã cập nhật schema bảng tickets.")
            except Exception as e:
                print('ensure_tickets_schema ALTER failed:', e)
        cur.close()
        db.close()
    except Exception as e:
        print('ensure_tickets_schema error:', e)


def user_has_active_monthly(user_id):
    """Kiểm tra user có vé tháng còn hạn không."""
    try:
        db = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM tickets WHERE user_id=%s AND ticket_type='monthly' AND status='NEW'", (user_id,))
        rows = cur.fetchall()
        cur.close()
        db.close()
        for ex in rows:
            valid_from = ex.get('valid_from') or ex.get('purchase_time')
            if valid_from:
                try:
                    vf = valid_from if isinstance(valid_from, (datetime, date)) else datetime.fromisoformat(str(valid_from)).date()
                    if (datetime.now().date() - vf).days < 30:
                        return True
                except Exception:
                    continue
        return False
    except Exception as e:
        print('user_has_active_monthly error:', e)
        return False


def user_has_face(user_id):
    """Kiểm tra user đã đăng ký khuôn mặt chưa."""
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute('SELECT 1 FROM face_data WHERE user_id=%s LIMIT 1', (user_id,))
        r = cur.fetchone()
        cur.close()
        db.close()
        return bool(r)
    except Exception as e:
        print('user_has_face error:', e)
        return False


def init_admin():
    """Tạo tài khoản admin nếu chưa có (Đã cập nhật mật khẩu mã hóa)."""
    try:
        db = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute('SELECT * FROM users WHERE username=%s', ('admin',))
        if not cur.fetchone():
            admin_password = hash_pw('admin123') 
            cur.execute(
                'INSERT INTO users (username, email, phone, password, role, balance) VALUES (%s, %s, %s, %s, %s, %s)',
                ('admin', 'admin@metro.local', '0000000000', admin_password, 'admin', 1000000)
            )
            db.commit()
            print("Đã tạo tài khoản admin mặc định.")
        cur.close()
        db.close()
    except Exception as e:
        print(f'Lỗi khởi tạo admin: {e}')


def is_admin():
    """Kiểm tra session có phải là admin không."""
    if 'user' not in session:
        return False
    db = get_db()
    cur = db.cursor(dictionary=True)
    try:
        cur.execute('SELECT role FROM users WHERE id=%s', (session['user']['id'],))
        user = cur.fetchone()
        return user and user.get('role') == 'admin'
    finally:
        cur.close()
        db.close()

# ===================================================================
# ✨ ROUTES CƠ BẢN (AUTH, HOME)
# ===================================================================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        phone = request.form.get('phone', '').strip()
        email = request.form.get('email', '').strip()
        
        # ✨ CẬP NHẬT: Thêm .strip() cho mật khẩu
        password = request.form.get('password', '').strip()
        password_confirm = request.form.get('password_confirm', '').strip()
        
        # --- LOGIC AN TOÀN ---
        role = 'user' 
        is_student_checkbox = request.form.get('is_student')
        db_user_type = 'student' if is_student_checkbox == 'true' else 'general'
        # --- Hết logic an toàn ---

        if not all([username, phone, email, password, password_confirm]):
            return render_template('register.html', error='Vui lòng điền tất cả các trường')
       
        if password != password_confirm:
            return render_template('register.html', error='Mật khẩu không khớp')
       
        if len(password) < 6:
            return render_template('register.html', error='Mật khẩu phải có ít nhất 6 ký tự')
       
        password_hash = hash_pw(password) # Hash mật khẩu đã strip
        db = get_db()
        cur = db.cursor()
        try:
            cur.execute(
                'INSERT INTO users (username, phone, email, password, role, user_type) VALUES (%s, %s, %s, %s, %s, %s)',
                (username, phone, email, password_hash, role, db_user_type)
            )
            db.commit()
            return redirect(url_for('login'))
        except mysql.connector.Error as e:
            if 'Duplicate entry' in str(e):
                return render_template('register.html', error='Tên đăng nhập hoặc email đã tồn tại')
            return render_template('register.html', error=f'Lỗi đăng ký: {str(e)}')
        finally:
            cur.close()
            db.close()
    return render_template('register.html')


# ===================================================================
# ✨ HÀM LOGIN
# ===================================================================
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        
        # ✨ CẬP NHẬT: Thêm .strip() cho mật khẩu
        password_raw = request.form.get('password', '') # Mật khẩu gốc (có thể có dấu cách)
        password = password_raw.strip() # Mật khẩu đã xóa dấu cách
       
        if not username or not password: # Kiểm tra bằng mật khẩu đã xóa dấu cách
            return render_template('login.html', error='Vui lòng nhập tên đăng nhập và mật khẩu')
       
        password_hash = hash_pw(password) # Luôn mã hóa mật khẩu đã xóa dấu cách
        db = get_db()
        cur = db.cursor(dictionary=True)
        try:
            # 1. Thử đăng nhập bình thường (dùng mật khẩu đã mã hóa và xóa dấu cách)
            cur.execute('SELECT * FROM users WHERE username=%s AND password=%s', (username, password_hash))
            user = cur.fetchone()
            
            # 2. MẸO SỬA LỖI TỰ ĐỘNG (Dành cho admin)
            if not user and (username == 'admin' or 'admin' in username):
                cur.execute('SELECT * FROM users WHERE username=%s AND password=%s', (username, password_raw))
                user_dang_bi_loi = cur.fetchone()

                if not user_dang_bi_loi:
                    cur.execute('SELECT * FROM users WHERE username=%s AND password=%s', (username, password))
                    user_dang_bi_loi = cur.fetchone()
                
                if user_dang_bi_loi:
                    print(f"PHÁT HIỆN MẬT KHẨU ADMIN BỊ LỖI! TỰ ĐỘNG SỬA...")
                    cur.execute("UPDATE users SET password = %s WHERE username = %s", (password_hash, username))
                    db.commit()
                    print(f"ĐÃ SỬA MẬT KHẨU CHO '{username}'.")
                    user = user_dang_bi_loi
            # ✨ HẾT MẸO SỬA LỖI ✨

            if user:
                session['user'] = {
                    'id': user.get('id'),
                    'username': user.get('username'),
                    'email': user.get('email'),
                    'phone': user.get('phone'),
                    'role': user.get('role', 'user'),
                    'balance': user.get('balance', 0), 
                    'user_type': user.get('user_type', 'general') 
                }
                
                if user.get('role') == 'admin':
                    return render_template('open_admin.html')
                
                return redirect(url_for('home'))
            else:
                return render_template('login.html', error='Tên đăng nhập hoặc mật khẩu không đúng')
        except Exception as e:
            return render_template('login.html', error=f'Lỗi đăng nhập: {str(e)}')
        finally:
            cur.close()
            db.close()
    return render_template('login.html')
# ===================================================================


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))


# ===================================================================
# ✨ API VÍ VÀ GIÁ VÉ
# ===================================================================

@app.route('/api/get_stations', methods=['GET'])
def get_stations_api():
    try:
        stations = get_all_stations()
        return jsonify({'stations': stations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/calculate_price', methods=['POST'])
def calculate_price():
    if 'user' not in session:
        return jsonify({'error': 'Chưa đăng nhập'}), 403
    
    data = request.json
    from_station = data.get('from_station')
    to_station = data.get('to_station')
    
    if not from_station or not to_station:
        return jsonify({'error': 'Vui lòng chọn ga đi và ga đến'}), 400
    
    price = get_ticket_price(from_station, to_station)
    
    if price == -1:
        return jsonify({'error': 'Lỗi tính giá, ga không hợp lệ'}), 500
        
    return jsonify({'success': True, 'price': price})


@app.route('/api/wallet/topup', methods=['POST'])
def topup_wallet():
    """Mô phỏng nạp tiền vào ví."""
    if 'user' not in session:
        return jsonify({'error': 'Chưa đăng nhập'}), 403
    
    user_id = session['user']['id']
    amount = request.json.get('amount', 100000) 
    
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute("UPDATE users SET balance = balance + %s WHERE id = %s", (amount, user_id))
        cur.execute("INSERT INTO wallet_transactions (user_id, amount, type) VALUES (%s, %s, 'top-up')", (user_id, amount))
        db.commit()
        
        session['user']['balance'] = session['user'].get('balance', 0) + amount
        session.modified = True
        
        return jsonify({'success': True, 'new_balance': session['user']['balance']})
    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        db.close()

# ===================================================================
# ✨ GIAI ĐOẠN 3: API GỢI Ý (AI) LỊCH TÀU
# ===================================================================
@app.route('/api/get_travel_suggestion', methods=['POST'])
def get_travel_suggestion():
    if 'user' not in session:
        return jsonify({'error': 'Chưa đăng nhập'}), 403

    try:
        data = request.json
        valid_date_str = data.get('date')     # "2025-11-16"
        departure_time_str = data.get('time') # "10:00"
        
        if not valid_date_str or not departure_time_str:
            return jsonify({'error': 'Thiếu ngày hoặc giờ'}), 400
            
        departure_datetime = datetime.strptime(f"{valid_date_str} {departure_time_str}", '%Y-%m-%d %H:%M')
        
        # Gọi hàm logic "AI"
        frequency, suggestion = get_train_frequency(departure_datetime)
        
        return jsonify({'success': True, 'frequency': frequency, 'suggestion': suggestion})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===================================================================
# ✨ GIAI ĐOẠN 3: API GỢI Ý THÓI QUEN (MỚI)
# ===================================================================
@app.route('/api/get_user_habits')
def get_user_habits():
    if 'user' not in session:
        return jsonify({'error': 'Chưa đăng nhập'}), 403
    
    user_id = session['user']['id']
    
    db = get_db()
    cur = db.cursor(dictionary=True)
    try:
        # Tìm (ga đi, ga đến, giờ) phổ biến nhất cho vé lượt
        # Chúng ta group by GIỜ (HOUR) để làm tròn (ví dụ 8:01 và 8:15 đều là 8 giờ)
        cur.execute("""
            SELECT 
                from_station_name, 
                to_station_name, 
                HOUR(expected_departure_time) as habit_hour, 
                COUNT(*) as frequency
            FROM tickets
            WHERE 
                user_id = %s 
                AND ticket_type = 'single'
                AND from_station_name IS NOT NULL
                AND to_station_name IS NOT NULL
                AND expected_departure_time IS NOT NULL
            GROUP BY 
                from_station_name, 
                to_station_name, 
                habit_hour
            ORDER BY 
                frequency DESC
            LIMIT 1;
        """, (user_id,))
        
        habit = cur.fetchone()
        
        if habit:
            return jsonify({'success': True, 'habit': habit})
        else:
            return jsonify({'success': False, 'message': 'Không có thói quen nào được ghi nhận.'})
            
    except Exception as e:
        print(f"Lỗi get_user_habits: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        db.close()

# ===================================================================
# ✨ GIAI ĐOẠN 3: API CHATBOT (MỚI)
# ===================================================================
def parse_chat_query(query):
    """
    Hàm "AI" Rule-based đơn giản để phân tích
    tin nhắn của người dùng.
    """
    query_lower = query.lower()
    
    # --- 1. Phân tích Chào hỏi ---
    if re.search(r'\b(chào|xin chào|hello|hi)\b', query_lower):
        return "Chào bạn, tôi là trợ lý ảo của Metro FaceCheck. Tôi có thể giúp gì cho bạn? (Ví dụ: 'giá vé từ ... đến ...', 'lịch tàu lúc 8h sáng mai')"

    # --- 2. Phân tích Lịch tàu / Tần suất ---
    if re.search(r'lịch tàu|tần suất|mấy phút|bao lâu.*chuyến|ga (đông|vắng|thưa)', query_lower):
        
        # Tìm giờ (ví dụ: 8h, 9 giờ, 10:30)
        time_match = re.search(r'(\d+)(:|h| giờ)', query_lower)
        hour = int(time_match.group(1)) if time_match else datetime.now().hour
        
        # Tìm ngày
        target_date = datetime.now()
        if re.search(r'cuối tuần|thứ 7|chủ nhật', query_lower):
            # Giả định là Thứ 7 tới
            target_date = datetime.now() + timedelta(days=(5 - datetime.now().weekday() + 7) % 7)
        elif re.search(r'ngày mai', query_lower):
            target_date = datetime.now() + timedelta(days=1)
        
        # Tạo datetime object để phân tích
        try:
            analysis_dt = target_date.replace(hour=hour, minute=0)
            freq, suggestion = get_train_frequency(analysis_dt)
            return f"Vào khoảng {hour}h {target_date.strftime('%d/%m')}: {suggestion}"
        except Exception as e:
            return "Xin lỗi, tôi chưa hiểu rõ giờ bạn muốn hỏi. Vui lòng thử lại (ví dụ: 'tần suất tàu lúc 8h sáng')."

    # --- 3. Phân tích Giá vé ---
    if re.search(r'giá vé|bao nhiêu tiền|tốn.*tiền', query_lower):
        stations_found = []
        for station_name in STATIONS_LIST:
            # Tìm tên ga (loại bỏ chữ "Ga " để tìm kiếm linh hoạt hơn)
            if re.search(station_name.replace("Ga ", "").lower(), query_lower):
                stations_found.append(station_name)
        
        if len(stations_found) < 2:
            return "Tôi cần biết 2 ga (ga đi và ga đến) để tra giá vé. (Ví dụ: 'giá vé từ Bến Thành đến Suối Tiên')"
        
        try:
            # Giả định thứ tự là [từ, đến]
            s1, s2 = stations_found[0], stations_found[1]
            price = get_ticket_price(s1, s2)
            if price >= 0:
                return f"Giá vé lượt đi từ <strong>{s1}</strong> đến <strong>{s2}</strong> là {price:,.0f} VNĐ (đã giảm 1.000đ)."
            else:
                raise Exception("Lỗi tính giá")
        except Exception as e:
            return "Rất tiếc, tôi không thể tìm thấy giá vé cho 2 ga đó. Vui lòng kiểm tra lại tên ga."

    # --- 4. Fallback ---
    return "Xin lỗi, tôi chưa hiểu ý bạn. Bạn có thể hỏi tôi về <strong>giá vé</strong> hoặc <strong>lịch tàu</strong>."


@app.route('/api/chat', methods=['POST'])
def api_chat():
    if 'user' not in session:
        return jsonify({'error': 'Chưa đăng nhập'}), 403
        
    try:
        message = request.json.get('message', '').strip()
        if not message:
            return jsonify({'reply': 'Vui lòng nhập câu hỏi của bạn.'})
            
        # Gọi hàm "AI"
        reply_message = parse_chat_query(message)
        
        return jsonify({'reply': reply_message})
        
    except Exception as e:
        print(f"Lỗi API Chat: {e}")
        return jsonify({'reply': 'Đã có lỗi xảy ra, vui lòng thử lại.'}), 500
# ===================================================================


# ===================================================================
# ✨ GIAI ĐOẠN 1: CẬP NHẬT HÀM `buy_ticket`
# ===================================================================
@app.route('/buy_ticket', methods=['GET','POST'])
def buy_ticket():
    if 'user' not in session:
        return redirect(url_for('login'))
       
    # --- PHẦN XỬ LÝ CHUNG ---
    stations = get_all_stations()
    user_id = session['user']['id']
    balance = session.get('user', {}).get('balance', 0)
    user_type = session.get('user', {}).get('user_type', 'general')
    has_active = user_has_active_monthly(user_id)
    has_face = user_has_face(user_id)
   
    # --- XỬ LÝ GET ---
    if request.method == 'GET':
        return render_template('buy_ticket.html', 
                               stations=stations, 
                               has_active=has_active, 
                               has_face=has_face, 
                               balance=balance, 
                               user_type=user_type,
                               error=request.args.get('error'))

    # --- XỬ LÝ POST ---
    if request.method == 'POST':
        
        db = get_db()
        cur = db.cursor(dictionary=True)
        db.autocommit = False 
        
        cur.execute("SELECT balance, user_type FROM users WHERE id = %s FOR UPDATE", (user_id,))
        user_data = cur.fetchone()
        current_balance = user_data['balance']
        db_user_type = user_data['user_type']

        try:
            ticket_type = request.form.get('ticket_type') 
            price = 0
            
            cols = ['user_id', 'ticket_type', 'purchase_time', 'status', 'purchase_price', 'used', 'trip_code']
            vals = [user_id, ticket_type, datetime.now(), 'NEW', 0, 0, str(uuid.uuid4())]

            # --- 1. LOGIC CHO VÉ LƯỢT ---
            if ticket_type == 'single':
                from_station = request.form.get('from_station')
                to_station = request.form.get('to_station')
                
                valid_date_str = request.form.get('valid_date')     # "2025-11-16"
                departure_time_str = request.form.get('departure_time') # "10:00"
                
                if not all([from_station, to_station, valid_date_str, departure_time_str]):
                    return render_template('buy_ticket.html', error='Vé lượt cần đủ ga đi, ga đến, ngày và giờ đi.', stations=stations, balance=balance, user_type=user_type, has_active=has_active, has_face=has_face)
                
                if from_station == to_station:
                     return render_template('buy_ticket.html', error='Ga đi và ga đến phải khác nhau.', stations=stations, balance=balance, user_type=user_type, has_active=has_active, has_face=has_face)

                price = get_ticket_price(from_station, to_station)
                if price < 0: 
                     return render_template('buy_ticket.html', error='Lỗi tính giá, ga không hợp lệ.', stations=stations, balance=balance, user_type=user_type, has_active=has_active, has_face=has_face)
                
                try:
                    departure_datetime = datetime.strptime(f"{valid_date_str} {departure_time_str}", '%Y-%m-%d %H:%M')
                except ValueError:
                    return render_template('buy_ticket.html', error='Ngày hoặc giờ không hợp lệ.', stations=stations, balance=balance, user_type=user_type, has_active=has_active, has_face=has_face)

                cols.extend(['from_station_name', 'to_station_name', 'valid_from', 'expected_departure_time'])
                vals.extend([from_station, to_station, departure_datetime.date(), departure_datetime])

            # --- 2. LOGIC CHO VÉ THÁNG ---
            elif ticket_type == 'monthly':
                if has_active:
                    return render_template('buy_ticket.html', error='Bạn đã có vé tháng đang hoạt động.', stations=stations, balance=balance, user_type=user_type, has_active=has_active, has_face=has_face)

                price = 150000 if db_user_type == 'student' else 300000
                
                from_station_default = request.form.get('station_from_id') 
                to_station_default = request.form.get('station_to_id') 

                if not from_station_default or not to_station_default:
                     return render_template('buy_ticket.html', error='Vui lòng chọn ga mặc định.', stations=stations, balance=balance, user_type=user_type, has_active=has_active, has_face=has_face)

                cols.extend(['from_station_name', 'to_station_name', 'valid_from'])
                vals.extend([from_station_default, to_station_default, datetime.now().date()])

            else:
                return render_template('buy_ticket.html', error='Lỗi: Loại vé không hợp lệ.', stations=stations, balance=balance, user_type=user_type, has_active=has_active, has_face=has_face)

            # --- 3. LOGIC THANH TOÁN (GIAO DỊCH) ---
            if current_balance < price:
                return render_template('buy_ticket.html', error=f'Số dư không đủ. Cần {price:,.0f} VNĐ, bạn có {current_balance:,.0f} VNĐ.', stations=stations, balance=balance, user_type=user_type, has_active=has_active, has_face=has_face)
            
            vals[cols.index('purchase_price')] = price

            cur.execute("UPDATE users SET balance = balance - %s WHERE id = %s", (price, user_id))
            
            placeholders = ','.join(['%s'] * len(vals))
            q = f"INSERT INTO tickets ({', '.join(cols)}) VALUES ({placeholders})"
            cur.execute(q, tuple(vals))
            ticket_id = cur.lastrowid 
            
            cur.execute(
                "INSERT INTO wallet_transactions (user_id, amount, type, ticket_id) VALUES (%s, %s, 'purchase', %s)",
                (user_id, -price, ticket_id)
            )
            
            db.commit() 

            # --- 4. CẬP NHẬT SESSION VÀ CHUYỂN HƯỚNG ---
            session['user']['balance'] = current_balance - price
            session.modified = True
            
            if has_face:
                return redirect(url_for('history'))
            else:
                return redirect(url_for('upload_face_page'))
       
        except Exception as e:
            db.rollback() 
            traceback.print_exc()
            return render_template('buy_ticket.html', error=f'Lỗi giao dịch: {str(e)}', stations=stations, balance=balance, user_type=user_type, has_active=has_active, has_face=has_face)
        
        finally:
            db.autocommit = True 
            cur.close()
            db.close()


@app.route('/upload_face', methods=['GET'])
def upload_face_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('upload_face.html')


@app.route('/api/upload_face', methods=['POST'])
def upload_face():
    if 'user' not in session:
        return jsonify({'error': 'Chưa đăng nhập'}), 403
   
    try:
        file = request.files.get('face')
        if not file:
            return jsonify({'error': 'Không có tệp'}), 400
       
        username = session['user'].get('username')
        safe_name = ''.join(c for c in username if c.isalnum() or c in ('-', '_')).strip()
        user_id = session['user']['id']
        os.makedirs(config.FACES_DIR, exist_ok=True)
       
        file_extension = os.path.splitext(file.filename)[1] or '.jpg'
       
        for f in os.listdir(config.FACES_DIR):
            if f.startswith(safe_name + '.'):
                try:
                    os.remove(os.path.join(config.FACES_DIR, f))
                except Exception as e:
                    print(f"Không thể xóa file cũ: {e}")


        filename = os.path.join(config.FACES_DIR, f"{safe_name}{file_extension}")
        file.save(filename)
       
        db = get_db()
        cur = db.cursor()
        try:
            cur.execute('SELECT * FROM face_data WHERE user_id=%s', (user_id,))
            if cur.fetchone():
                cur.execute('UPDATE face_data SET photo_path=%s WHERE user_id=%s', (filename, user_id))
            else:
                cur.execute('INSERT INTO face_data (user_id, photo_path) VALUES (%s, %s)', (user_id, filename))
            db.commit()
            
            cur.execute('UPDATE users SET face_registered=1 WHERE id=%s', (user_id,))
            db.commit()
            
            load_known_faces() # Tải lại khuôn mặt sau khi upload
            
            return jsonify({'success': True, 'message': 'Đã tải lên khuôn mặt thành công'})
        except Exception as e:
            return jsonify({'error': f'Lỗi DB: {str(e)}'}), 500
        finally:
            cur.close()
            db.close()
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'}), 500


@app.route('/checkin', methods=['GET'])
def checkin_page():
    try:
        stations = get_all_stations()
    except Exception as e:
        stations = []
        print(f'Lỗi lấy danh sách ga: {e}')
    return render_template('checkin.html', stations=stations)


# ===================================================================
# ✨ API CHECK-IN (ĐÃ NÂNG CẤP LIVENESS 2.0)
# ===================================================================
@app.route('/api/checkin', methods=['POST'])
def api_checkin():
    try:
        data = request.json
        img_b64 = data.get('image_b64')
        station_name = data.get('station') 
       
        if not img_b64 or not station_name:
            return jsonify({'error': 'Thiếu dữ liệu image_b64 hoặc station'}), 400
       
        # 1. Xử lý ảnh đầu vào
        img_bytes = base64.b64decode(img_b64.split(',')[-1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_h, img_w, _ = img.shape

        # --- ✨ LIVENESS DETECTION 2.0 (MEDIAPIPE) ✨ ---
        # Chuyển đổi màu sang RGB cho MediaPipe
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return jsonify({
                'success': False, 
                'reason': 'no_face_detected', 
                'message': 'Không tìm thấy khuôn mặt. Vui lòng đứng trước camera.'
            })

        # Lấy khuôn mặt đầu tiên tìm thấy
        landmarks = results.multi_face_landmarks[0].landmark

        # 2. Kiểm tra Tỷ lệ mở mắt (EAR)
        # Nếu EAR thấp, người dùng có thể đang nhắm mắt hoặc dùng ảnh in
        left_ear = calculate_ear(landmarks, LEFT_EYE, img_w, img_h)
        right_ear = calculate_ear(landmarks, RIGHT_EYE, img_w, img_h)
        avg_ear = (left_ear + right_ear) / 2.0

        # Ngưỡng mở mắt (0.22 là ngưỡng thực nghiệm phổ biến)
        if avg_ear < 0.22:
            return jsonify({
                'success': False,
                'reason': 'eyes_closed',
                'message': 'Mắt bạn đang nhắm. Vui lòng mở mắt và nhìn thẳng.'
            })

        # 3. Kiểm tra Hướng đầu (Head Pose)
        # Đảm bảo người dùng nhìn thẳng vào camera (chống ảnh 2D nghiêng)
        is_straight = check_head_pose(landmarks, img_w, img_h)
        if not is_straight:
            return jsonify({
                'success': False,
                'reason': 'bad_pose',
                'message': 'Vui lòng nhìn THẲNG vào camera (đừng quay ngang).'
            })
        # --- ✨ HẾT PHẦN LIVENESS ✨ ---
       
        # 4. Nhận diện khuôn mặt (Face Recognition)
        # Nếu đã qua bước Liveness, tiến hành so khớp
        unknown_encodings = face_recognition.face_encodings(rgb_image)
        if not unknown_encodings:
            return jsonify({'success': False, 'reason': 'no_face_encoding', 'message': 'Ảnh mờ hoặc không rõ nét.'})
       
        unknown_enc = unknown_encodings[0]
       
        if not known_face_data:
             return jsonify({'success': False, 'reason': 'no_known_faces', 'message': 'Hệ thống chưa có dữ liệu khuôn mặt.'})
       
        known_usernames = [data[0] for data in known_face_data]
        known_encodings = [data[1] for data in known_face_data]

        distances = face_recognition.face_distance(known_encodings, unknown_enc)
        best_match_index = np.argmin(distances)
        matched_username = ""
       
        if distances[best_match_index] < config.FACE_RECOGNITION_TOLERANCE: 
            matched_username = known_usernames[best_match_index]

        if not matched_username:
            # Ghi log thất bại (Không tìm thấy người)
            db = get_db(); cur = db.cursor()
            try:
                cur.execute('''
                    INSERT INTO checkins (station, checkin_time, success, user_id, ticket_id)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (station_name, datetime.now(), 0, None, None))
                db.commit()
            finally:
                cur.close(); db.close()
            return jsonify({'success': False, 'reason': 'no_match', 'message': 'Không tìm thấy vé đăng ký với khuôn mặt này.'})
       
        # 5. Kiểm tra logic vé (Người dùng đã được xác định)
        db = get_db()
        cur = db.cursor(dictionary=True)
        try:
            cur.execute("SELECT id FROM users WHERE username=%s", (matched_username,))
            user_row = cur.fetchone()
           
            if not user_row:
                return jsonify({'success': False, 'reason': 'user_not_found', 'message': f'Khuôn mặt khớp với {matched_username} nhưng không tìm thấy user trong CSDL'})
           
            matched_user_id = user_row['id']
            now = datetime.now()
           
            # Chặn check-in liên tục trong 5 phút
            cur.execute("SELECT station, checkin_time FROM checkins WHERE user_id = %s AND success = 1 ORDER BY checkin_time DESC LIMIT 1", (matched_user_id,))
            last_log = cur.fetchone()
           
            if last_log:
                last_time = last_log['checkin_time']
                last_station = last_log['station']
                time_diff_seconds = (now - last_time).total_seconds()
               
                if time_diff_seconds < 300: # 5 phút
                    cur.execute('''
                        INSERT INTO checkins (ticket_id, user_id, station, checkin_time, success)
                        VALUES (%s, %s, %s, %s, %s)
                    ''', (None, matched_user_id, station_name, now, 0)) 
                    db.commit()
                    return jsonify({
                        'success': False,
                        'user_id': matched_user_id,
                        'reason': 'rapid_checkin_denied',
                        'message': f'LỖI: Bạn vừa check-in tại "{last_station}" {int(time_diff_seconds // 60)} phút {int(time_diff_seconds % 60)} giây trước.'
                    })

            # Tìm vé hợp lệ
            cur.execute('''
                SELECT * FROM tickets
                WHERE user_id=%s AND status='NEW'
                ORDER BY ticket_type DESC, purchase_time ASC 
            ''', (matched_user_id,))
            
            tickets = cur.fetchall()
            today = now.date()

            if not tickets:
                cur.execute('''
                    INSERT INTO checkins (user_id, station, checkin_time, success)
                    VALUES (%s, %s, %s, %s)
                ''', (matched_user_id, station_name, now, 0))
                db.commit()
                return jsonify({'success': False, 'reason': 'no_ticket', 'message': 'Không tìm thấy vé nào còn hiệu lực'})

            for ticket in tickets:
                ticket_id = ticket['id']
                
                # --- XỬ LÝ VÉ LƯỢT (SINGLE) ---
                if ticket['ticket_type'] == 'single':
                    valid_date = ticket['valid_from']
                    
                    # 1. Kiểm tra ngày
                    if valid_date != today:
                        continue 
                        
                    # 2. Kiểm tra ga
                    if ticket['from_station_name'] != station_name:
                        cur.execute('''
                            INSERT INTO checkins (ticket_id, user_id, station, checkin_time, success)
                            VALUES (%s, %s, %s, %s, %s)
                        ''', (ticket_id, matched_user_id, station_name, now, 0))
                        db.commit()
                        return jsonify({'success': False, 'reason': 'wrong_station', 'message': f"SAI GA. Vé của bạn là từ: {ticket['from_station_name']}"})
                    
                    # 3. Kiểm tra giờ hợp lệ (+/- 30 phút)
                    expected_time = ticket['expected_departure_time']
                    if expected_time: 
                        grace_period = timedelta(minutes=30)
                        window_start = expected_time - grace_period
                        window_end = expected_time + grace_period
                        
                        if not (window_start <= now <= window_end):
                            cur.execute('''
                                INSERT INTO checkins (ticket_id, user_id, station, checkin_time, success)
                                VALUES (%s, %s, %s, %s, %s)
                            ''', (ticket_id, matched_user_id, station_name, now, 0))
                            db.commit()
                            return jsonify({
                                'success': False, 
                                'reason': 'wrong_time', 
                                'message': f"VÉ SAI GIỜ. Vé của bạn chỉ hợp lệ từ {window_start.strftime('%H:%M')} đến {window_end.strftime('%H:%M')}."
                            })
                    
                    # THÀNH CÔNG VÉ LƯỢT
                    cur.execute("UPDATE tickets SET status='USED', used=1 WHERE id=%s", (ticket_id,))
                    cur.execute('''
                        INSERT INTO checkins (ticket_id, user_id, station, checkin_time, success)
                        VALUES (%s, %s, %s, %s, %s)
                    ''', (ticket_id, matched_user_id, station_name, now, 1))
                    db.commit()
                    return jsonify({'success': True, 'reason': 'single_ok', 'message': 'Vé lượt hợp lệ. Mời vào.'})
                
                # --- XỬ LÝ VÉ THÁNG (MONTHLY) ---
                elif ticket['ticket_type'] == 'monthly':
                    valid_from_date = (ticket.get('valid_from') or ticket.get('purchase_time'))
                    if isinstance(valid_from_date, datetime):
                        valid_from_date = valid_from_date.date()
                    
                    if (today - valid_from_date).days >= 30:
                        continue 

                    # THÀNH CÔNG VÉ THÁNG
                    cur.execute('''
                        INSERT INTO checkins (ticket_id, user_id, station, checkin_time, success)
                        VALUES (%s, %s, %s, %s, %s)
                    ''', (ticket_id, matched_user_id, station_name, now, 1))
                    db.commit()
                    return jsonify({'success': True, 'reason': 'monthly_ok', 'message': 'Vé tháng hợp lệ. Mời vào.'})

            # Nếu vòng lặp chạy hết mà không return -> Không có vé hợp lệ
            cur.execute('''
                INSERT INTO checkins (user_id, station, checkin_time, success)
                VALUES (%s, %s, %s, %s)
            ''', (matched_user_id, station_name, now, 0))
            db.commit()
            return jsonify({'success': False, 'reason': 'all_tickets_invalid', 'message': 'Vé của bạn đã hết hạn hoặc không hợp lệ cho ngày/ga này.'})

        finally:
            cur.close()
            db.close()
   
    except Exception as e:
        print('Lỗi nghiêm trọng trong api_checkin:', e)
        traceback.print_exc()
        return jsonify({'error': str(e), 'message': 'Lỗi server nội bộ. Vui lòng kiểm tra log.'}), 500


@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))
   
    user_id = session['user']['id']
    db = get_db()
    cur = db.cursor(dictionary=True)
    try:
        cur.execute('''
            SELECT
                t.id, t.status, t.purchase_time,
                t.valid_from, t.expected_departure_time, 
                t.from_station_name, t.to_station_name,
                t.ticket_type, t.trip_code, t.used, t.purchase_price
            FROM tickets t
            WHERE t.user_id=%s
            ORDER BY t.purchase_time DESC
        ''', (user_id,))
        tickets = cur.fetchall()
    except Exception as e:
        tickets = []
        print(f'Lỗi lấy lịch sử: {e}')
    finally:
        cur.close()
        db.close()
   
    return render_template('history.html', tickets=tickets)


# ===================================================================
# ✨ ROUTE TRANG VÍ
# ===================================================================
@app.route('/wallet')
def wallet_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user']['id']
    
    db = get_db()
    cur = db.cursor(dictionary=True)
    try:
        cur.execute("SELECT balance FROM users WHERE id = %s", (user_id,))
        user_data = cur.fetchone()
        balance = user_data.get('balance', 0)
        
        session['user']['balance'] = balance
        session.modified = True
        
        cur.execute("""
            SELECT 
                t.amount, t.type, t.transaction_time, 
                tk.ticket_type, tk.from_station_name, tk.to_station_name
            FROM wallet_transactions t
            LEFT JOIN tickets tk ON t.ticket_id = tk.id
            WHERE t.user_id = %s
            ORDER BY t.transaction_time DESC
        """, (user_id,))
        transactions = cur.fetchall()
        
    except Exception as e:
        print(f"Lỗi tải trang ví: {e}")
        transactions = []
        balance = session.get('user', {}).get('balance', 0) 
    finally:
        cur.close()
        db.close()

    return render_template('wallet.html', balance=balance, transactions=transactions)
# ===================================================================


# ===================================================================
# ✨ GIAI ĐOẠN 3: ROUTE TRANG THÔNG BÁO (MỚI)
# ===================================================================
@app.route('/notifications')
def notifications_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user']['id']
    now = datetime.now()
    
    db = get_db()
    cur = db.cursor(dictionary=True)
    try:
        # Lấy tất cả vé lượt, CHƯA SỬ DỤNG, bắt đầu từ BÂY GIỜ
        # (và trong vòng 24 giờ tới để tránh quá tải)
        time_limit = now + timedelta(hours=24)
        
        cur.execute("""
            SELECT * FROM tickets
            WHERE 
                user_id = %s 
                AND ticket_type = 'single' 
                AND status = 'NEW'
                AND expected_departure_time >= %s
                AND expected_departure_time <= %s
            ORDER BY expected_departure_time ASC
        """, (user_id, now, time_limit))
        upcoming_tickets = cur.fetchall()
        
    except Exception as e:
        print(f"Lỗi tải thông báo: {e}")
        upcoming_tickets = []
    finally:
        cur.close()
        db.close()

    # Truyền 'now' vào template để JS tính toán thời gian còn lại
    return render_template('notifications.html', 
                           upcoming_tickets=upcoming_tickets, 
                           current_time_iso=now.isoformat())
# ===================================================================


@app.route('/admin')
def admin():
    if 'user' not in session:
        return redirect(url_for('login'))
   
    if not is_admin():
        return render_template('admin_error.html', message='Bạn không có quyền truy cập'), 403
   
    db = get_db()
    cur = db.cursor(dictionary=True)
    try:
        cur.execute('SELECT COUNT(*) as total_users FROM users WHERE role="user"')
        total_users = cur.fetchone()['total_users']
       
        cur.execute('SELECT COUNT(*) as total_tickets FROM tickets')
        total_tickets = cur.fetchone()['total_tickets']
       
        cur.execute('SELECT COUNT(*) as total_checkins FROM checkins WHERE success=1')
        total_checkins = cur.fetchone()['total_checkins']
       
        cur.execute('SELECT SUM(purchase_price) as total_revenue FROM tickets WHERE status="NEW" OR used=1')
        total_revenue = cur.fetchone()['total_revenue'] or 0
       
        # ✨ CẬP NHẬT ADMIN: Lấy thêm balance, user_type
        cur.execute('SELECT id, username, email, phone, face_registered, balance, user_type FROM users WHERE role="user" ORDER BY id DESC')
        users = cur.fetchall()
       
        # ✨ CẬP NHẬT ADMIN: Lấy thêm dữ liệu vé
        cur.execute('''
            SELECT t.*,
                   u.username
            FROM tickets t
            LEFT JOIN users u ON t.user_id = u.id
            ORDER BY t.purchase_time DESC
        ''')
        tickets = cur.fetchall()
       
        cur.execute('''
            SELECT ci.id as log_id, ci.checkin_time as timestamp, ci.success, ci.station as station_name,
                   u.username
            FROM checkins ci
            LEFT JOIN users u ON ci.user_id = u.id
            ORDER BY ci.checkin_time DESC
            LIMIT 100
        ''')
        entry_logs = cur.fetchall()
       
        cur.execute('''
            SELECT
                station as station_name,
                COUNT(*) as total_checkins,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_checkins
            FROM checkins
            WHERE station IS NOT NULL AND station != '' AND station != '0'
            GROUP BY station
            ORDER BY total_checkins DESC
        ''')
        station_stats = cur.fetchall()
       
    except Exception as e:
        print(f'Lỗi admin: {e}')
        traceback.print_exc()
        return render_template('admin_error.html', message=f'Lỗi tải dữ liệu: {str(e)}'), 500
    finally:
        cur.close()
        db.close()
   
    return render_template('admin.html',
                          total_users=total_users,
                          total_tickets=total_tickets,
                          total_checkins=total_checkins,
                          total_revenue=total_revenue,
                          users=users,
                          tickets=tickets,
                          entry_logs=entry_logs,
                          station_stats=station_stats 
                          )


@app.route('/api/admin/user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    if 'user' not in session or not is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
   
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute('DELETE FROM users WHERE id=%s AND role="user"', (user_id,))
        db.commit()
        return jsonify({'success': True, 'message': 'Xóa người dùng thành công'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        db.close()


@app.route('/api/admin/ticket/<int:ticket_id>', methods=['DELETE'])
def delete_ticket(ticket_id):
    if 'user' not in session or not is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
   
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute('DELETE FROM tickets WHERE id=%s', (ticket_id,))
        db.commit()
        return jsonify({'success': True, 'message': 'Xóa vé thành công'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        db.close()


# ===================================================================
# ✨ CẬP NHẬT ADMIN: Sửa hàm 'edit_user'
# ===================================================================
@app.route('/api/admin/user/<int:user_id>', methods=['PUT'])
def edit_user(user_id):
    if 'user' not in session or not is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
   
    try:
        data = request.json
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        phone = data.get('phone', '').strip()
        balance = data.get('balance', 0) 
        user_type = data.get('user_type', 'general') 
       
        if not all([username, email, phone]):
            return jsonify({'error': 'Vui lòng điền tất cả các trường'}), 400
        
        try:
            balance = int(balance)
        except ValueError:
            return jsonify({'error': 'Số dư phải là một con số nguyên'}), 400

        if user_type not in ['general', 'student']:
            return jsonify({'error': 'Loại người dùng không hợp lệ (phải là general hoặc student)'}), 400

        db = get_db()
        cur = db.cursor()
        try:
            cur.execute(
                'UPDATE users SET username=%s, email=%s, phone=%s, balance=%s, user_type=%s WHERE id=%s AND role="user"',
                (username, email, phone, balance, user_type, user_id)
            )
            db.commit()
            return jsonify({'success': True, 'message': 'Cập nhật người dùng thành công'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            cur.close()
            db.close()
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'}), 500


@app.route('/api/admin/stats', methods=['GET'])
def get_stats():
    if 'user' not in session or not is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
   
    db = get_db()
    cur = db.cursor(dictionary=True)
    try:
        cur.execute('SELECT COUNT(*) as total_users FROM users WHERE role="user"')
        total_users = cur.fetchone()['total_users']
       
        cur.execute('SELECT COUNT(*) as total_tickets FROM tickets')
        total_tickets = cur.fetchone()['total_tickets']
       
        cur.execute('SELECT COUNT(*) as today_checkins FROM checkins WHERE DATE(checkin_time)=DATE(NOW()) AND success=1')
        today_checkins = cur.fetchone()['today_checkins']
       
        return jsonify({
            'total_users': total_users,
            'total_tickets': total_tickets,
            'today_checkins': today_checkins
        })
    finally:
        cur.close()
        db.close()


@app.route('/api/admin/station_checkins/<path:station_name>')
def get_station_checkins(station_name):
    if 'user' not in session or not is_admin():
        return jsonify({'error': 'Unauthorized'}), 403

    decoded_station_name = urllib.parse.unquote(station_name)

    db = get_db()
    cur = db.cursor(dictionary=True)
    try:
        cur.execute('''
            SELECT 
                ci.id as log_id, 
                ci.checkin_time, 
                u.username,
                t.trip_code
            FROM checkins ci
            LEFT JOIN users u ON ci.user_id = u.id
            LEFT JOIN tickets t ON ci.ticket_id = t.id
            WHERE ci.station = %s AND ci.success = 1
            ORDER BY ci.checkin_time DESC
        ''', (decoded_station_name,))
        
        checkins = cur.fetchall()
        
        for checkin in checkins:
            if checkin['checkin_time']:
                checkin['checkin_time'] = checkin['checkin_time'].strftime('%d/%m/%Y %H:%M:%S')

        return jsonify({'success': True, 'checkins': checkins})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        db.close()

if __name__ == '__main__':
    os.makedirs(config.FACES_DIR, exist_ok=True)
    init_admin()
    
    try:
        ensure_stations(STATIONS_LIST)
    except Exception as e:
        print('Warning: ensure_stations failed:', e)
    try:
        ensure_face_data_table()
    except Exception as e:
        print('Warning: ensure_face_data_table failed:', e)
    try:
        ensure_tickets_schema()
    except Exception as e:
        print('Warning: ensure_tickets_schema failed:', e)
   
    try:
        load_known_faces()
    except Exception as e:
        print('Warning: load_known_faces failed:', e)


    app.run(host='0.0.0.0', port=5000, debug=True)