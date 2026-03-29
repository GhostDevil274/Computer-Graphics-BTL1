import numpy as np
import math

def translate(x, y, z):
    """ Dời vật thể """
    mat = np.identity(4, dtype=np.float32)
    mat[0, 3] = x
    mat[1, 3] = y
    mat[2, 3] = z
    return mat

def scale(sx, sy, sz):
    """ Phóng to/thu nhỏ (Hỗ trợ scale từng trục riêng biệt) """
    mat = np.identity(4, dtype=np.float32)
    mat[0, 0] = sx
    mat[1, 1] = sy
    mat[2, 2] = sz
    return mat

def rotate_y(angle_degrees):
    """ Xoay vật thể quanh trục Y (Trái/Phải) """
    rad = math.radians(angle_degrees)
    c = math.cos(rad)
    s = math.sin(rad)
    mat = np.identity(4, dtype=np.float32)
    mat[0, 0] = c;  mat[0, 2] = s
    mat[2, 0] = -s; mat[2, 2] = c
    return mat

def rotate_x(angle_degrees):
    """ Xoay vật thể quanh trục X (Cúi/Ngửa) """
    rad = math.radians(angle_degrees)
    c = math.cos(rad)
    s = math.sin(rad)
    mat = np.identity(4, dtype=np.float32)
    mat[1, 1] = c;  mat[1, 2] = -s
    mat[2, 1] = s;  mat[2, 2] = c
    return mat

def look_at(eye, target, up):
    """ Ma trận Camera """
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    res = np.identity(4, dtype=np.float32)
    res[0, :3] = s; res[1, :3] = u; res[2, :3] = -f
    res[0, 3] = -np.dot(s, eye)
    res[1, 3] = -np.dot(u, eye)
    res[2, 3] = np.dot(f, eye)
    return res

def perspective(fovy, aspect, zNear, zFar):
    """ Ma trận Ống kính """
    f = 1.0 / math.tan(math.radians(fovy) / 2.0)
    res = np.zeros((4,4), dtype=np.float32)
    res[0,0] = f / aspect
    res[1,1] = f
    res[2,2] = (zFar + zNear) / (zNear - zFar)
    res[2,3] = (2.0 * zFar * zNear) / (zNear - zFar)
    res[3,2] = -1.0
    return res

class Trackball:
    """ Lớp Camera quỹ đạo (Orbit Camera) điều khiển bằng chuột 3D """
    def __init__(self, distance=4.0, target=(0.0, 0.0, 0.0)):
        self.distance = distance
        self.azimuth = 0.0     # Góc xoay quanh trục Y (trái/phải)
        self.elevation = 20.0  # Góc nhìn từ trên xuống (cúi/ngửa)
        self.target = np.array(target, dtype=np.float32)

    def drag(self, old, new, win_size):
        """ Cập nhật góc xoay khi kéo chuột trái """
        dx = new[0] - old[0]
        dy = new[1] - old[1]
        
        # Tốc độ xoay (Sensitivity)
        self.azimuth -= dx * 0.5
        self.elevation += dy * 0.5
        
        # Khóa góc cúi/ngửa (-89 đến 89 độ) để camera không bị lộn ngược vòng
        self.elevation = max(-89.0, min(89.0, self.elevation))

    def pan(self, old, new):
        """ Cập nhật điểm nhìn (target) khi kéo chuột phải """
        dx = new[0] - old[0]
        dy = new[1] - old[1]
        
        # Tốc độ dịch chuyển phụ thuộc vào việc camera đang ở gần hay xa
        pan_speed = 0.002 * self.distance
        
        # Tính toán trục Ngang (Right) của Camera để dịch chuyển cho chuẩn
        az_rad = math.radians(self.azimuth)
        right_x = math.cos(az_rad)
        right_z = -math.sin(az_rad)
        
        # Dịch chuyển tâm nhìn
        self.target[0] -= (dx * right_x) * pan_speed
        self.target[1] -= dy * pan_speed             # Dịch lên xuống theo trục Y thế giới
        self.target[2] -= (dx * right_z) * pan_speed

    def zoom(self, delta, win_height=None):
        """ Phóng to / Thu nhỏ khi lăn chuột giữa """
        self.distance -= delta * 0.5
        
        # Không cho phép camera chui xuyên qua tâm vật thể (giới hạn min = 0.1)
        if self.distance < 0.1:
            self.distance = 0.1

    def view_matrix(self):
        """ Xuất ra Ma trận View cho file main.py """
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        
        # Dùng Lượng giác không gian (Spherical to Cartesian) để tính vị trí con mắt (Eye)
        cx = self.target[0] + self.distance * math.cos(el) * math.sin(az)
        cy = self.target[1] + self.distance * math.sin(el)
        cz = self.target[2] + self.distance * math.cos(el) * math.cos(az)
        
        eye = np.array([cx, cy, cz], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32) # Hướng lên trời mặc định
        
        # Tái sử dụng hàm look_at ở bên trên
        return look_at(eye, self.target, up)

    def projection_matrix(self, win_size):
        """ Xuất ra Ma trận Ống kính cho file main.py """
        aspect = win_size[0] / win_size[1] if win_size[1] > 0 else 1.0
        # Tái sử dụng hàm perspective ở bên trên
        return perspective(45.0, aspect, 0.1, 100.0)