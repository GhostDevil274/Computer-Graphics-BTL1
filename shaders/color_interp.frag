#version 330 core
in vec3 vColor;
out vec4 FragColor;

// Công tắc từ Python: 0 = Vẽ màu bình thường, 1 = Vẽ Depth Map
uniform int is_depth_map; 

void main() {
    if (is_depth_map == 1) {
        // Lấy thông số chiều sâu Z của đồ họa (chạy từ 0.0 đến 1.0)
        float depth = gl_FragCoord.z;
        // Bơm tọa độ Z vào cả 3 kênh R, G, B để tạo ra màu xám Trắng/Đen
        FragColor = vec4(depth, depth, depth, 1.0);
    } else {
        // Vẽ màu Cầu vồng bình thường
        FragColor = vec4(vColor, 1.0);
    }
}