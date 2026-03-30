#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec3 vertexColor; 
in vec2 TexCoord;

uniform int render_mode; 
uniform vec3 flat_color; 
uniform sampler2D tex_diffuse;

uniform vec3 viewPos;
uniform bool is_depth_map; 
uniform vec3 bg_color;

uniform bool light1_on;
uniform bool light2_on;
uniform bool light3_on; 

vec3 sunLightDir = normalize(vec3(0.5, 1.0, 0.5)); 
vec3 sunLightColor = vec3(1.4, 1.4, 1.4); 

vec3 pointLight1Pos = vec3(1.5, 0.0, 2.0); 
vec3 pointLight1Color = vec3(1.0, 0.5, 0.0);

vec3 pointLight2Pos = vec3(-1.5, 0.0, 2.0); 
vec3 pointLight2Color = vec3(0.0, 0.5, 1.0);

float near = 0.1;  
float far  = 50.0; 

float LinearizeDepth(float depth) 
{
    float z = depth * 2.0 - 1.0; 
    return (2.0 * near * far) / (far + near - z * (far - near));    
}

vec3 calcDirLight(vec3 lightDir, vec3 lightColor, vec3 norm, vec3 viewDir, vec3 baseColor) {
    // Tia sáng đã chuẩn
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor * baseColor;
    
    // Reflect yêu cầu tia tới hướng VÀO vật thể (-lightDir)
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = 0.5 * spec * lightColor;
    
    return diffuse + specular;
}

// Hàm tính Đèn Điểm
vec3 calcPointLight(vec3 lightPos, vec3 lightColor, vec3 norm, vec3 fragPos, vec3 viewDir, vec3 baseColor) {
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor * baseColor;
    
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = 0.5 * spec * lightColor;
    
    float distance = length(lightPos - fragPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * (distance * distance));
    
    return (diffuse + specular) * attenuation;
}

void main() {
    if (is_depth_map) {
        float depth = LinearizeDepth(gl_FragCoord.z) / far; 
        depth = clamp(depth, 0.0, 1.0);
        
        vec3 depthFogColor = mix(vec3(1.0), bg_color, depth);
        
        FragColor = vec4(depthFogColor, 1.0);
        return; 
    }

    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    float ambientStrength = 0.2;
    
    vec3 baseCol;
    if (render_mode == 0) {
        baseCol = flat_color; 
    } 
    else if (render_mode == 1 || render_mode == 2) {
        baseCol = vertexColor; 
    } 
    else if (render_mode == 3 || render_mode == 4) {
        baseCol = texture(tex_diffuse, TexCoord).rgb; 
    }

    vec3 final_color = ambientStrength * baseCol;

    if (render_mode == 0 || render_mode == 1 || render_mode == 3) {
        // (A), (B), (D) KHÔNG NHẬN ÁNH SÁNG (Unlit)
        // Lưu ý: (B) giờ là cầu vồng nhưng tối thui (chỉ có Ambient)
        FragColor = vec4(baseCol, 1.0); 
    } 
    else if (render_mode == 2 || render_mode == 4) {
        // (C), (E) BẬT ĐÈN LÊN VÀ TỎA SÁNG!
        // (C) bây giờ là cầu vồng lóng lánh ánh đèn!
        if (light1_on) final_color += calcDirLight(sunLightDir, sunLightColor, norm, viewDir, baseCol);
        if (light2_on) final_color += calcPointLight(pointLight1Pos, pointLight1Color, norm, FragPos, viewDir, baseCol);
        if (light3_on) final_color += calcPointLight(pointLight2Pos, pointLight2Color, norm, FragPos, viewDir, baseCol);
        
        FragColor = vec4(final_color, 1.0);
    }
}