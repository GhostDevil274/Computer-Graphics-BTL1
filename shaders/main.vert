#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;  // Pháp tuyến (Dùng cho ánh sáng)
layout (location = 3) in vec2 aTexCoord;// Tọa độ 2D (Dùng để dán ảnh Texture)

out vec3 FragPos;
out vec3 Normal;
out vec3 vertexColor;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    // Tính toán vị trí thực tế của vật thể trong thế giới để hứng ánh sáng
    FragPos = vec3(model * vec4(aPos, 1.0));
    // Xoay vector pháp tuyến theo vật thể
    Normal = mat3(transpose(inverse(model))) * aNormal;  
    
    vertexColor = aColor;
    TexCoord = aTexCoord;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}