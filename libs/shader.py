import OpenGL.GL as GL
import OpenGL.GL.shaders as shaders

class Shader:
    def __init__(self, vert_path, frag_path):
        self.render_idx = self._compile_shader(vert_path, frag_path)

    def _compile_shader(self, vert_path, frag_path):
        """ Đọc file text và biên dịch thành Shader Program """
        # Đọc nội dung file Vertex Shader
        with open(vert_path, 'r', encoding='utf-8') as f:
            vert_code = f.read()
        
        # Đọc nội dung file Fragment Shader
        with open(frag_path, 'r', encoding='utf-8') as f:
            frag_code = f.read()

        # Dùng thư viện OpenGL biên dịch
        vert_shader = shaders.compileShader(vert_code, GL.GL_VERTEX_SHADER)
        frag_shader = shaders.compileShader(frag_code, GL.GL_FRAGMENT_SHADER)
        
        # Gộp 2 thợ (Thợ Đỉnh và Thợ Màu) lại thành 1 Tổ đội (Program)
        shader_program = shaders.compileProgram(vert_shader, frag_shader)
        return shader_program

    def use(self):
        """ Gọi tổ đội này ra làm việc """
        GL.glUseProgram(self.render_idx)