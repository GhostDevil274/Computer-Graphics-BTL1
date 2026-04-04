import glfw
import OpenGL.GL as GL
import imgui
from imgui.integrations.glfw import GlfwRenderer
import numpy as np
import os
from PIL import Image

from gui import AppGUI
from libs.shader import Shader
from libs.transform import Trackball, scale, rotate_x, rotate_y, translate
from shapes.basic_2d import RegularPolygon, Rectangle, Ellipse, Trapezoid, Star, Arrow
from shapes.basic_3d import (
    Cylinder, Cone, TruncatedCone, Torus, SphereLatLong, 
    SphereSubdivision, SphereCube, Cube, Tetrahedron, MathSurface, ObjModel, PlyModel, HeatmapSurface
)
from libs.ai_optim import GradientDescent, Momentum, Nesterov, RMSprop, Adam, LossFunction

class SceneObject:
    def __init__(self, shape, name):
        self.shape = shape
        self.name = name
        self.scale = 1.0
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        
        self.render_mode = 1 
        self.flat_color = [0.8, 0.2, 0.3]
        self.texture_id = 0
        self.texture_filepath = "texture.jpeg"

def load_texture(filepath):
    if not filepath.startswith("assets"): filepath = os.path.join("assets", "textures", filepath)
    if not os.path.exists(filepath): return 0
    try:
        img = Image.open(filepath).transpose(Image.FLIP_TOP_BOTTOM)
        img = img.convert("RGBA") 
        img_data = np.array(img, np.uint8) 
        tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        return tex_id
    except: return 0

class LinePath:
    def __init__(self):
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        
    def draw(self, history, color, shader_idx, model_matrix, base_z_lift):
        if len(history) < 2: return
        
        # Add a tiny base lift so it doesn't clip into the grid
        pts = [[x, y, z_loss + base_z_lift] for (x, y, z_loss) in history]
        pts_array = np.array(pts, dtype=np.float32)
        
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, pts_array.nbytes, pts_array, GL.GL_DYNAMIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)
        
        GL.glUniform1i(GL.glGetUniformLocation(shader_idx, "render_mode"), 0)
        GL.glUniform3f(GL.glGetUniformLocation(shader_idx, "flat_color"), *color)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader_idx, "model"), 1, GL.GL_TRUE, model_matrix)
        
        GL.glLineWidth(4.0) 
        GL.glDrawArrays(GL.GL_LINE_STRIP, 0, len(pts))
        GL.glLineWidth(1.0) 
            
        GL.glBindVertexArray(0)

def screen_to_world_ray(xpos, ypos, win_w, win_h, view_matrix, proj_matrix):
    ndc_x = (2.0 * xpos) / win_w - 1.0
    ndc_y = 1.0 - (2.0 * ypos) / win_h
    ray_clip = np.array([ndc_x, ndc_y, -1.0, 1.0])
    inv_proj = np.linalg.inv(proj_matrix)
    ray_eye = np.dot(inv_proj, ray_clip)
    ray_eye = np.array([ray_eye[0], ray_eye[1], -1.0, 0.0])
    inv_view = np.linalg.inv(view_matrix)
    ray_wor = np.dot(inv_view, ray_eye)[:3]
    ray_wor = ray_wor / np.linalg.norm(ray_wor)
    cam_pos = inv_view[:3, 3]
    return cam_pos, ray_wor

def main():
    if not glfw.init(): return
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4) 
    
    monitor = glfw.get_primary_monitor()
    video_mode = glfw.get_video_mode(monitor)
    win_w = int(video_mode.size.width * 0.75)
    win_h = int(video_mode.size.height * 0.75)
    
    window = glfw.create_window(win_w, win_h, "Viewer", None, None)
    glfw.set_window_pos(window, int((video_mode.size.width - win_w)/2), int((video_mode.size.height - win_h)/2))

    glfw.make_context_current(window)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_MULTISAMPLE) 
    
    imgui.create_context()
    io = imgui.get_io()
    io.font_global_scale = 1.36
    imgui.style_colors_dark() 
    impl = GlfwRenderer(window)
    
    dummy_vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(dummy_vao)
    
    my_shader = Shader("shaders/main.vert", "shaders/main.frag")
    
    dummy_tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, dummy_tex)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, 1, 1, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, np.array([255, 255, 255, 255], dtype=np.uint8))
    
    line_drawer = LinePath()
    agent_sphere = SphereSubdivision(radius=1.0)
    
    scene_objects = []
    ai_shape = None
    gui = AppGUI()
    current_epoch = 0
    
    optims = [
        GradientDescent("GD/SGD", [1.0, 0.2, 0.2], 0, 0),
        Momentum("Momentum", [0.2, 1.0, 0.2], 0, 0),
        Nesterov("Nesterov", [1.0, 0.8, 0.0], 0, 0),
        RMSprop("RMSprop", [0.8, 0.3, 1.0], 0, 0), 
        Adam("Adam", [0.2, 0.5, 1.0], 0, 0)
    ]

    cameras = [Trackball(distance=15.0), Trackball(distance=20.0), Trackball(distance=10.0)]
    cameras[1].elevation, cameras[1].azimuth = 80.0, 0.0 
    cameras[2].elevation, cameras[2].azimuth = 20.0, -45.0

    ball_tex = load_texture("ball.jpg")

    def set_race_track(loss_name, keep_pos=False):
        nonlocal ai_shape, current_epoch
        current_epoch = 0
        
        if not keep_pos:
            gui.selected_cam_idx = 1 
            gui.ai_rot_x, gui.ai_rot_y = -90.0, 0.0 
            cameras[1].target = [0.0, 0.0, 0.0]
            
            ai_shape = HeatmapSurface(loss_name=loss_name, custom_func_str=gui.custom_loss_str)
            
            if loss_name == "Quadratic": 
                gui.custom_z_scale = 0.15; gui.start_x = -5.0; gui.start_y = 5.0
                gui.ai_scale = 1.0
                cameras[1].distance, cameras[1].elevation, cameras[1].azimuth = 13.0, 45.0, 45.0
            elif loss_name == "Booth": 
                gui.custom_z_scale = 0.005; gui.start_x = -8.0; gui.start_y = 8.0
                gui.ai_scale = 0.7
                cameras[1].distance, cameras[1].elevation, cameras[1].azimuth = 18.0, 55.0, 30.0
            elif loss_name == "Himmelblau": 
                gui.custom_z_scale = 0.01; gui.start_x = -5.0; gui.start_y = 5.0
                gui.ai_scale = 0.8
                cameras[1].distance, cameras[1].elevation, cameras[1].azimuth = 15.0, 55.0, -20.0
            elif loss_name == "Rosenbrock": 
                gui.custom_z_scale = 0.002; gui.start_x = -2.5; gui.start_y = 2.5
                gui.ai_scale = 1.8   
                cameras[1].distance, cameras[1].elevation, cameras[1].azimuth = 14.0, 45.0, 50.0
            elif loss_name == "Custom":
                gui.ai_scale = 1.0   
                cameras[1].distance, cameras[1].elevation, cameras[1].azimuth = 15.0, 45.0, 45.0

        micro_offsets = [-0.2, -0.1, 0.0, 0.1, 0.2] 
        for i, opt in enumerate(optims):
            opt.reset(gui.start_x + micro_offsets[i], gui.start_y)
            opt.z, _, _ = LossFunction.get_val_and_grad(loss_name, opt.x, opt.y, gui.custom_loss_str)

    def on_key(win, key, scancode, action, mods):
        impl.keyboard_callback(win, key, scancode, action, mods)
        if imgui.get_io().want_capture_keyboard: return
        if action == glfw.PRESS:
            if key == glfw.KEY_1: gui.lights[0] = not gui.lights[0]
            elif key == glfw.KEY_2: gui.lights[1] = not gui.lights[1]
            elif key == glfw.KEY_3: gui.lights[2] = not gui.lights[2]
    glfw.set_key_callback(window, on_key)

    def on_mouse_click(win, button, action, mods):
        if imgui.get_io().want_capture_mouse: return
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and not gui.is_ai_mode:
            if mods & glfw.MOD_SHIFT or mods & glfw.MOD_CONTROL: return
            
            xpos, ypos = glfw.get_cursor_pos(win)
            ww, wh = glfw.get_window_size(win)
            cur_cam = cameras[gui.selected_cam_idx]
            v_mat = cur_cam.view_matrix()
            p_mat = cur_cam.projection_matrix((ww, wh))
            
            ray_orig, ray_dir = screen_to_world_ray(xpos, ypos, ww, wh, v_mat, p_mat)
            
            closest_dist = float('inf')
            selected_idx = -1
            
            for i, obj in enumerate(scene_objects):
                obj_pos = np.array([obj.pos_x, obj.pos_y, obj.pos_z])
                vec = obj_pos - ray_orig
                t = np.dot(vec, ray_dir)
                if t > 0: 
                    proj_point = ray_orig + t * ray_dir
                    dist = np.linalg.norm(obj_pos - proj_point)
                    hit_radius = 1.0 * obj.scale 
                    if dist < hit_radius and t < closest_dist:
                        closest_dist = t
                        selected_idx = i
                        
            if selected_idx != -1:
                gui.selected_scene_obj_idx = selected_idx
    glfw.set_mouse_button_callback(window, on_mouse_click)

    def on_mouse_move(win, xpos, ypos):
        if not hasattr(on_mouse_move, "old_pos"): on_mouse_move.old_pos = (xpos, ypos)
        
        old_x, old_y = on_mouse_move.old_pos
        dx = xpos - old_x
        dy = ypos - old_y 
        on_mouse_move.old_pos = (xpos, ypos)
        
        if imgui.get_io().want_capture_mouse: return
        
        is_shift = glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        has_obj = len(scene_objects) > 0 and not gui.is_ai_mode
        
        rot_sens = 0.2
        pan_sens = 0.015
        
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            if is_shift and has_obj:
                active_obj = scene_objects[gui.selected_scene_obj_idx]
                active_obj.rot_y += dx * rot_sens
                active_obj.rot_x += dy * rot_sens 
            else:
                cameras[gui.selected_cam_idx].azimuth -= dx * rot_sens
                cameras[gui.selected_cam_idx].elevation = max(-89.0, min(89.0, cameras[gui.selected_cam_idx].elevation - dy * rot_sens))
                
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            if is_shift and has_obj:
                active_obj = scene_objects[gui.selected_scene_obj_idx]
                cam_az = np.radians(cameras[gui.selected_cam_idx].azimuth)
                right_x = np.cos(cam_az)
                right_z = np.sin(cam_az)
                active_obj.pos_x += dx * pan_sens * right_x
                active_obj.pos_z += dx * pan_sens * right_z
                active_obj.pos_y -= dy * pan_sens 
            else:
                cameras[gui.selected_cam_idx].pan((xpos, ypos), (old_x, old_y))
                gui.spawn_pos = list(cameras[gui.selected_cam_idx].target)
                
    glfw.set_cursor_pos_callback(window, on_mouse_move)

    def on_scroll(win, dx, dy):
        impl.scroll_callback(win, dx, dy)
        if not imgui.get_io().want_capture_mouse: 
            is_ctrl = glfw.get_key(win, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
            if is_ctrl and len(scene_objects) > 0 and not gui.is_ai_mode:
                active_obj = scene_objects[gui.selected_scene_obj_idx]
                active_obj.scale = max(0.1, active_obj.scale + dy * 0.1)
            else:
                cameras[gui.selected_cam_idx].zoom(dy, glfw.get_window_size(win)[1])
    glfw.set_scroll_callback(window, on_scroll)

    last_step_time = glfw.get_time()

    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        glfw.poll_events()
        impl.process_inputs()

        if getattr(gui, 'texture_changed', False):
            if len(scene_objects) > 0:
                target_obj = scene_objects[getattr(gui, 'target_tex_obj_idx', 0)]
                filepath = target_obj.texture_filepath
                
                if not filepath.startswith("assets"):
                    if os.path.exists(os.path.join("assets", "models", filepath)):
                        actual_img_path = os.path.join("assets", "models", filepath)
                    else:
                        actual_img_path = os.path.join("assets", "textures", filepath)
                else:
                    actual_img_path = filepath

                print(f"⏳ Đang nạp ảnh trực tiếp từ: {actual_img_path} ...")
                tex_id = load_texture(actual_img_path)
                
                if tex_id > 0:
                    if target_obj.texture_id > 0: 
                        GL.glDeleteTextures(1, [target_obj.texture_id])
                    target_obj.texture_id = tex_id
                    target_obj.render_mode = 4
                    print("✅ Dán ảnh THÀNH CÔNG! Nhanh gọn lẹ!")
                else:
                    print(f"❌ [LỖI] Không tìm thấy ảnh. Ông gõ lại đúng tên file ảnh giùm tui cái!")
                    
            gui.texture_changed = False

        if getattr(gui, 'delete_obj_requested', False):
            if len(scene_objects) > 0:
                scene_objects.pop(gui.selected_scene_obj_idx)
                gui.selected_scene_obj_idx = max(0, gui.selected_scene_obj_idx - 1) 
            gui.delete_obj_requested = False
            
        if getattr(gui, 'duplicate_obj_requested', False):
            if len(scene_objects) > 0:
                old_obj = scene_objects[gui.selected_scene_obj_idx]
                new_obj = SceneObject(old_obj.shape, old_obj.name + " (Copy)")
                new_obj.scale, new_obj.rot_x, new_obj.rot_y = old_obj.scale, old_obj.rot_x, old_obj.rot_y
                new_obj.pos_x, new_obj.pos_y, new_obj.pos_z = old_obj.pos_x + 2.0, old_obj.pos_y, old_obj.pos_z
                new_obj.render_mode, new_obj.texture_id = old_obj.render_mode, old_obj.texture_id
                new_obj.texture_filepath = old_obj.texture_filepath
                new_obj.flat_color = list(old_obj.flat_color)
                scene_objects.append(new_obj)
                gui.selected_scene_obj_idx = len(scene_objects) - 1
            gui.duplicate_obj_requested = False

        if gui.reset_requested:
            if gui.is_ai_mode: set_race_track(gui.loss_funcs[gui.selected_loss_idx], keep_pos=False)
            gui.reset_requested = False
            gui.sim_playing = False

        if gui.reposition_requested:
            if gui.is_ai_mode: set_race_track(gui.loss_funcs[gui.selected_loss_idx], keep_pos=True)
            gui.reposition_requested = False
            gui.sim_playing = False

        if gui.add_shape_requested:
            idx = gui.selected_shape_idx
            new_shape = None
            shape_name = gui.shape_names[idx].split(". ")[-1]
            
            if idx == 1: new_shape = RegularPolygon(sides=3)
            elif idx == 2: new_shape = Rectangle()
            elif idx == 3: new_shape = RegularPolygon(sides=5)
            elif idx == 4: new_shape = RegularPolygon(sides=6)
            elif idx == 5: new_shape = RegularPolygon(sides=36)
            elif idx == 6: new_shape = Ellipse()
            elif idx == 7: new_shape = Trapezoid()
            elif idx == 8: new_shape = Star()
            elif idx == 9: new_shape = Arrow()
            elif idx == 10: new_shape = Cylinder()
            elif idx == 11: new_shape = Cone()
            elif idx == 12: new_shape = TruncatedCone()
            elif idx == 13: new_shape = Torus()
            elif idx == 14: new_shape = SphereLatLong()
            elif idx == 15: new_shape = SphereSubdivision() 
            elif idx == 16: new_shape = SphereCube()       
            elif idx == 17: new_shape = Cube()               
            elif idx == 18: new_shape = Tetrahedron()        
            elif idx == 19: new_shape = Cylinder(segments=6)
            elif idx == 20: 
                new_shape = MathSurface(func_str=gui.math_func_str)
                shape_name = "Math Surface"
            elif idx == 21: 
                obj_path = gui.obj_filepath
                if not obj_path.startswith("assets"): obj_path = os.path.join("assets", "models", obj_path)
                
                if obj_path.lower().endswith(".ply"):
                    new_shape = PlyModel(filepath=obj_path)
                    shape_name = "PLY Model"
                else:
                    new_shape = ObjModel(filepath=obj_path)
                    shape_name = "OBJ Model"
                
            if new_shape is not None:
                obj = SceneObject(new_shape, f"#{len(scene_objects)+1} {shape_name}")
                if hasattr(new_shape, 'texture_file') and new_shape.texture_file:
                    obj.texture_filepath = new_shape.texture_file
                    tex_id = load_texture(new_shape.texture_file)
                    if tex_id > 0:
                        obj.texture_id = tex_id
                        obj.render_mode = 4
                elif hasattr(new_shape, 'diffuse_color') and new_shape.diffuse_color:
                    obj.flat_color = new_shape.diffuse_color
                    obj.render_mode = 3

                obj.pos_x, obj.pos_y, obj.pos_z = gui.spawn_pos
                scene_objects.append(obj)
                gui.selected_scene_obj_idx = len(scene_objects) - 1
                gui.spawn_pos[0] += 2.5 
                
            gui.add_shape_requested = False
            
        if gui.clear_scene_requested:
            scene_objects = []
            gui.clear_scene_requested = False
            gui.selected_scene_obj_idx = 0

        if gui.loss_changed and gui.is_ai_mode:
            loss_name = gui.loss_funcs[gui.selected_loss_idx]
            if loss_name == "Custom" and not gui.force_load_ai:
                ai_shape = None; gui.sim_playing = False
            else:
                set_race_track(loss_name, keep_pos=False)
            gui.loss_changed = False
            gui.force_load_ai = False

        optims[1].beta = gui.momentum_beta 

        if gui.is_ai_mode and getattr(gui, 'sim_playing', False) and current_epoch < gui.max_epochs and ai_shape is not None:
            time_per_step = 1.0 / gui.steps_per_sec
            if current_time - last_step_time >= time_per_step:
                actual_radius = 0.18 * gui.ai_scale 
                
                for i, opt in enumerate(optims):
                    if gui.opt_active[i]:
                        if not hasattr(opt, 'last_pos'): opt.last_pos = np.array([opt.x, opt.y])
                        if not hasattr(opt, 'rot_mat'): opt.rot_mat = np.identity(4)
                        
                        opt.step(gui.loss_funcs[gui.selected_loss_idx], gui.lr, noise_level=getattr(gui, 'noise_level', 0.0), custom_func_str=gui.custom_loss_str)
                        
                        dx = opt.x - opt.last_pos[0]
                        dy = opt.y - opt.last_pos[1]
                        
                        if np.hypot(dx, dy) > 0.0001:
                            angle_factor = (1.0 / actual_radius) * (180.0 / np.pi) * (gui.roll_speed / 300.0)
                            
                            r_x = rotate_x(-dy * angle_factor)
                            r_y = rotate_y(dx * angle_factor)

                            opt.rot_mat = np.matmul(np.matmul(r_x, r_y), opt.rot_mat)
                            opt.last_pos = np.array([opt.x, opt.y])
                            
                current_epoch += 1
                last_step_time = current_time
                if current_epoch >= gui.max_epochs: 
                    gui.sim_playing = False

        gui.render(optims, current_epoch, cameras, scene_objects)
        
        if gui.is_ai_mode:
            if not hasattr(gui, 'ball_offset_z'): gui.ball_offset_z = 0.0
            if not hasattr(gui, 'ball_offset_x'): gui.ball_offset_x = 0.0
            if not hasattr(gui, 'ball_offset_y'): gui.ball_offset_y = 0.0
            if not hasattr(gui, 'roll_speed'): gui.roll_speed = 300.0

            imgui.begin("AI Ball Physics Tweaker", closable=False)
            imgui.text_colored("Coordinate Tweaks (Anti-Clipping):", 0.2, 1.0, 0.2)
            _, gui.ball_offset_z = imgui.slider_float("Z Offset (Up/Down)", gui.ball_offset_z, -0.5, 1.0)
            _, gui.ball_offset_x = imgui.slider_float("X Offset (Left/Right)", gui.ball_offset_x, -1.0, 1.0)
            _, gui.ball_offset_y = imgui.slider_float("Y Offset (Forward/Back)", gui.ball_offset_y, -1.0, 1.0)
            imgui.spacing()
            _, gui.roll_speed = imgui.slider_float("Roll Speed", gui.roll_speed, 10.0, 1000.0)
            
            if imgui.button("Reset Offsets"):
                gui.ball_offset_z = 0.0
                gui.ball_offset_x = 0.0
                gui.ball_offset_y = 0.0
            imgui.end()

        fb_width, fb_height = glfw.get_framebuffer_size(window)
        win_size = glfw.get_window_size(window)
        current_cam = cameras[gui.selected_cam_idx]
        view_matrix = current_cam.view_matrix()
        projection_matrix = current_cam.projection_matrix(win_size)
        
        GL.glViewport(0, 0, fb_width, fb_height)
        
        active_depth_map = False if gui.is_ai_mode else gui.is_depth_map
        GL.glClearColor(*gui.bg_color, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        my_shader.use()
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(my_shader.render_idx, "view"), 1, GL.GL_TRUE, view_matrix)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(my_shader.render_idx, "projection"), 1, GL.GL_TRUE, projection_matrix)
        
        GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "is_depth_map"), active_depth_map)
        GL.glUniform3f(GL.glGetUniformLocation(my_shader.render_idx, "bg_color"), *gui.bg_color)
        
        GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "light1_on"), gui.lights[0])
        GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "light2_on"), gui.lights[1])
        GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "light3_on"), gui.lights[2])
        inv_view = np.linalg.inv(view_matrix)
        GL.glUniform3f(GL.glGetUniformLocation(my_shader.render_idx, "viewPos"), *inv_view[:3, 3])

        if getattr(gui, 'is_wireframe', False): 
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        else: 
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        if gui.is_ai_mode:
            if ai_shape is not None:
                m_trans = translate(0.0, 0.0, 0.0)
                m_scale = scale(gui.ai_scale, gui.ai_scale, gui.ai_scale * gui.custom_z_scale)
                m_rot_x = rotate_x(gui.ai_rot_x)
                m_rot_y = rotate_y(gui.ai_rot_y)
                ai_matrix = np.matmul(m_trans, np.matmul(np.matmul(m_rot_y, m_rot_x), m_scale))
                
                if not getattr(gui, 'is_wireframe', False):
                    GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
                    GL.glPolygonOffset(1.0, 1.0)
                GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "render_mode"), 1)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(my_shader.render_idx, "model"), 1, GL.GL_TRUE, ai_matrix)
                ai_shape.draw()
                
                if not getattr(gui, 'is_wireframe', False):
                    GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
                    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
                    GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "render_mode"), 0)
                    GL.glUniform3f(GL.glGetUniformLocation(my_shader.render_idx, "flat_color"), 0.15, 0.15, 0.15)
                    ai_shape.draw()
                    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL) 

                offsets_x = [-0.04, -0.02, 0.0, 0.02, 0.04] 
                loss_name = gui.loss_funcs[gui.selected_loss_idx]

                for i, opt in enumerate(optims):
                    if gui.opt_active[i]:
                        
                        base_lift = 0.05 / gui.custom_z_scale if gui.custom_z_scale != 0 else 0.05
                        
                        visual_shift = translate(gui.ball_offset_x, gui.ball_offset_y, gui.ball_offset_z)
                        shifted_ai_matrix = np.matmul(ai_matrix, visual_shift)
                        
                        line_drawer.draw(opt.history, opt.color, my_shader.render_idx, shifted_ai_matrix, base_lift)
                        
                        actual_radius = 0.18 * gui.ai_scale 
                        
                        true_z, _, _ = LossFunction.get_val_and_grad(loss_name, opt.x, opt.y, gui.custom_loss_str)
                        local_pos = np.array([opt.x, opt.y, true_z + base_lift], dtype=np.float32)
                        
                        world_pos = np.matmul(shifted_ai_matrix, np.array([local_pos[0], local_pos[1], local_pos[2], 1.0]))
                        
                        bi_trans_world = translate(world_pos[0], world_pos[1] + actual_radius, world_pos[2])
                        bi_scale_world = scale(actual_radius, actual_radius, actual_radius)
                        
                        if not hasattr(opt, 'rot_mat'): opt.rot_mat = np.identity(4)
                        bi_model = np.matmul(bi_trans_world, np.matmul(opt.rot_mat, bi_scale_world))
                        
                        GL.glUniformMatrix4fv(GL.glGetUniformLocation(my_shader.render_idx, "model"), 1, GL.GL_TRUE, bi_model)
                        
                        if 'ball_tex' in locals() and ball_tex > 0:
                            GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "render_mode"), 5) 
                            GL.glActiveTexture(GL.GL_TEXTURE0)
                            GL.glBindTexture(GL.GL_TEXTURE_2D, ball_tex)
                            GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "tex_diffuse"), 0)
                        else:
                            GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "render_mode"), 0)
                            
                        GL.glUniform3f(GL.glGetUniformLocation(my_shader.render_idx, "flat_color"), *opt.color)
                        agent_sphere.draw()
        else:
            for obj in scene_objects:
                m_trans = translate(obj.pos_x, obj.pos_y, obj.pos_z)
                m_scale = scale(obj.scale, obj.scale, obj.scale)
                m_rot_x = rotate_x(obj.rot_x)
                m_rot_y = rotate_y(obj.rot_y)
                model_matrix = np.matmul(m_trans, np.matmul(np.matmul(m_rot_y, m_rot_x), m_scale))
                
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(my_shader.render_idx, "model"), 1, GL.GL_TRUE, model_matrix)
                
                GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "render_mode"), obj.render_mode)
                GL.glUniform3f(GL.glGetUniformLocation(my_shader.render_idx, "flat_color"), *obj.flat_color)
                
                if obj.texture_id > 0 and obj.render_mode in [3, 4] and not active_depth_map:
                    GL.glActiveTexture(GL.GL_TEXTURE0)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, obj.texture_id)
                    GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "tex_diffuse"), 0)
                
                obj.shape.draw()
            
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()