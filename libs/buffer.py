import OpenGL.GL as GL
import numpy as np
import ctypes

class VAO:
    def __init__(self):
        # Tạo 1 cái Thùng Hàng Lớn (Vertex Array Object)
        self.vao = GL.glGenVertexArrays(1)
        self.vbos = [] # Danh sách các hộp nhỏ bên trong
        self.ebo = None

    def add_vbo(self, location, data, ncomponents=3):
        """ 
        Đóng gói Dữ liệu (Tọa độ, Màu sắc, Pháp tuyến) vào hộp VBO 
        location: 0 (Position), 1 (Color), 2 (Normal)
        """
        GL.glBindVertexArray(self.vao)
        
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        # Ném mảng numpy xuống GPU
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        
        # Ghi nhãn dán hướng dẫn GPU cách đọc hộp này
        GL.glVertexAttribPointer(location, ncomponents, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(location)
        
        self.vbos.append(vbo)
        GL.glBindVertexArray(0) # Đóng thùng lại

    def add_ebo(self, indices):
        """ Đóng gói Bản đồ nối điểm (Indices) vào hộp EBO """
        GL.glBindVertexArray(self.vao)
        
        self.ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
        
        GL.glBindVertexArray(0)

    def activate(self):
        """ Mở thùng hàng ra chuẩn bị vẽ """
        GL.glBindVertexArray(self.vao)