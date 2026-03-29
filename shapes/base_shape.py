import OpenGL.GL as GL
import numpy as np
import math

class BaseShape:
    def __init__(self, vertices, indices, colors, normals=None, uvs=None):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)
        self.colors = np.array(colors, dtype=np.float32)
        
        # 1. TÍNH PHÁP TUYẾN ÁNH SÁNG 
        if normals is None:
            self.normals = np.zeros_like(self.vertices)
            for i in range(0, len(self.indices), 3):
                i1, i2, i3 = self.indices[i], self.indices[i+1], self.indices[i+2]
                v1, v2, v3 = self.vertices[i1], self.vertices[i2], self.vertices[i3]
                normal = np.cross(v2 - v1, v3 - v1)
                self.normals[i1] += normal
                self.normals[i2] += normal
                self.normals[i3] += normal
            lengths = np.linalg.norm(self.normals, axis=1, keepdims=True)
            lengths[lengths == 0] = 1.0
            self.normals /= lengths
        else:
            self.normals = np.array(normals, dtype=np.float32)

        # 2. TỌA ĐỘ DÁN ẢNH 
        if uvs is None:
            self.uvs = np.zeros((len(self.vertices), 2), dtype=np.float32)
            for i, v in enumerate(self.vertices):
                norm_v = v / (np.linalg.norm(v) + 1e-6)
                u = 0.5 + math.atan2(norm_v[2], norm_v[0]) / (2 * math.pi)
                v_tex = 0.5 - math.asin(norm_v[1]) / math.pi
                self.uvs[i] = [u, v_tex]
        else:
            self.uvs = np.array(uvs, dtype=np.float32)

        # 3. NẠP VÀO CARD ĐỒ HỌA
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        self.vbo_pos = self.create_vbo(0, self.vertices)
        self.vbo_col = self.create_vbo(1, self.colors)
        self.vbo_norm = self.create_vbo(2, self.normals)
        self.vbo_uv = self.create_vbo(3, self.uvs)

        self.ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL.GL_STATIC_DRAW)
        GL.glBindVertexArray(0)

    def create_vbo(self, location, data):
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(location, data.shape[1] if len(data.shape) > 1 else 1, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(location)
        return vbo

    def draw(self):
        GL.glBindVertexArray(self.vao)
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)