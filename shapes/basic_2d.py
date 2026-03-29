import numpy as np
import math
from shapes.base_shape import BaseShape

def generate_rainbow_colors(vertices):
    """ Hàm tạo màu cầu vồng 3D siêu mượt dựa trên tọa độ """
    coords = np.array(vertices, dtype=np.float32)
    max_val = np.max(np.abs(coords)) if np.max(np.abs(coords)) != 0 else 1.0
    # Ép tọa độ từ khoảng [-max, max] về [0.0, 1.0] để làm mã màu RGB
    return ((coords / max_val + 1.0) / 2.0).astype(np.float32)

class RegularPolygon(BaseShape):
    def __init__(self, sides=36, radius=0.5):
        vertices = [[0.0, 0.0, 0.0]]
        indices = []
        angle_step = (2 * math.pi) / sides
        for i in range(sides):
            vertices.append([radius * math.cos(i * angle_step), radius * math.sin(i * angle_step), 0.0])
        for i in range(1, sides + 1):
            indices.extend([0, i, 1 if i == sides else i + 1])
        super().__init__(np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), generate_rainbow_colors(vertices))

class Rectangle(BaseShape):
    def __init__(self):
        vertices = [[-0.5,-0.3,0], [0.5,-0.3,0], [0.5,0.3,0], [-0.5,0.3,0]]
        indices = [0, 1, 2, 0, 2, 3]
        super().__init__(np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), generate_rainbow_colors(vertices))

class Trapezoid(BaseShape):
    def __init__(self):
        vertices = [[-0.6,-0.4,0], [0.6,-0.4,0], [0.3,0.4,0], [-0.3,0.4,0]]
        indices = [0, 1, 2, 0, 2, 3]
        super().__init__(np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), generate_rainbow_colors(vertices))

class Ellipse(BaseShape):
    def __init__(self, a=0.6, b=0.3, segments=36):
        vertices = [[0.0, 0.0, 0.0]]
        indices = []
        angle_step = (2 * math.pi) / segments
        for i in range(segments):
            vertices.append([a * math.cos(i * angle_step), b * math.sin(i * angle_step), 0.0])
        for i in range(1, segments + 1):
            indices.extend([0, i, 1 if i == segments else i + 1])
        super().__init__(np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), generate_rainbow_colors(vertices))

class Star(BaseShape):
    def __init__(self, points=5, inner_r=0.2, outer_r=0.6):
        vertices = [[0.0, 0.0, 0.0]]
        indices = []
        angle_step = math.pi / points
        for i in range(points * 2):
            r = outer_r if i % 2 == 0 else inner_r
            a = i * angle_step + math.pi/2
            vertices.append([r * math.cos(a), r * math.sin(a), 0.0])
        for i in range(1, points * 2 + 1):
            indices.extend([0, i, 1 if i == points * 2 else i + 1])
        super().__init__(np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), generate_rainbow_colors(vertices))

class Arrow(BaseShape):
    def __init__(self):
        vertices = [
            [-0.5, -0.2, 0], [0.2, -0.2, 0], [0.2, -0.5, 0], 
            [0.8, 0.0, 0], [0.2, 0.5, 0], [0.2, 0.2, 0], [-0.5, 0.2, 0]
        ]
        indices = [0, 1, 5,  0, 5, 6,  2, 3, 4]
        super().__init__(np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), generate_rainbow_colors(vertices))