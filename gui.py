import imgui

class AppGUI:
    def __init__(self):
        self.active_tab = 0 
        
        self.selected_shape_idx = 0 
        self.shape_names = [
            "0. (Clear Screen)", "1. 2D Triangle", "2. 2D Rectangle", "3. 2D Pentagon", 
            "4. 2D Regular Hexagon", "5. 2D Circle", "6. 2D Ellipse", "7. 2D Trapezoid", 
            "8. 2D Star", "9. 2D Arrow", "10. 3D Cylinder", "11. 3D Cone", "12. 3D Truncated Cone", 
            "13. 3D Torus", "14. 3D Sphere (Lat-Long)", "15. 3D Sphere (Subdivision)", 
            "16. 3D Sphere (Norm Cube)", "17. 3D Cube", "18. 3D Tetrahedron", 
            "19. 3D Prism (Hexagonal)", "20. 3D Math Surface", "21. 3D OBJ Model"            
        ]
        self.shape_changed = True
        self.force_load_shape = False 
        
        self.math_func_str = "sin(x) + cos(y)"
        self.obj_filepath = "bugatti.obj"
        
        self.spawn_pos = [0.0, 0.0, 0.0]
        self.fetch_cam_target = False 
        self.add_shape_requested = False 
        self.clear_scene_requested = False
        
        self.delete_obj_requested = False
        self.duplicate_obj_requested = False
        self.selected_scene_obj_idx = 0 
        self.target_tex_obj_idx = 0
        self.texture_changed = False
        
        self.is_depth_map = False
        self.is_wireframe = False
        self.lights = [True, False, False]
        self.bg_color = [0.1, 0.1, 0.15]
        
        self.selected_cam_idx = 0

        self.ai_scale = 1.0
        self.ai_rot_x = -90.0
        self.ai_rot_y = 0.0

        self.is_ai_mode = False 
        self.loss_funcs = ["Quadratic", "Booth", "Himmelblau", "Rosenbrock", "Custom"]
        self.selected_loss_idx = 0
        self.loss_changed = False
        self.force_load_ai = False 
        
        self.custom_loss_str = "x**2 + y**2"
        self.custom_z_scale = 0.15  
        self.start_x = -3.0        
        self.start_y = 3.0         
        
        self.opt_active = [True, True, True, True, True]
        self.noise_level = 0.0
        self.lr = 0.005
        self.momentum_beta = 0.90
        self.max_epochs = 2000
        self.steps_per_sec = 10 
        self.sim_playing = False
        self.reset_requested = False
        self.reposition_requested = False 

    def render(self, optims, current_epoch, cameras, scene_objects):
        imgui.new_frame()
        imgui.set_next_window_size(420, 860, imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(20, 20, imgui.FIRST_USE_EVER)
        
        imgui.begin("SCENE & PROPERTIES", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        
        if self.is_ai_mode:
            imgui.text_colored("[!] AI MODE IS ACTIVE", 1.0, 0.3, 0.3)
            imgui.text("Disable AI Mode on the right panel\nto edit 3D Geometry.")
            imgui.spacing(); imgui.separator(); imgui.spacing()
            
        expanded_geo, _ = imgui.collapsing_header("1. GEOMETRY", flags=imgui.TREE_NODE_DEFAULT_OPEN)
        if expanded_geo and not self.is_ai_mode:
            _, self.selected_shape_idx = imgui.listbox("##ShapeList", self.selected_shape_idx, self.shape_names, 10)
            
            if self.selected_shape_idx == 20:
                imgui.text_colored("Custom Surface Equation:", 1.0, 0.8, 0.2)
                _, self.math_func_str = imgui.input_text("Z=f(x,y)", self.math_func_str, 256)
            elif self.selected_shape_idx == 21:
                imgui.text_colored("3D Model Asset:", 1.0, 0.8, 0.2)
                _, self.obj_filepath = imgui.input_text(".OBJ File", self.obj_filepath, 256)
            
            imgui.spacing()
            if self.selected_shape_idx > 0:
                imgui.text_colored("Spawn Location (Auto-follows Camera):", 0.8, 0.8, 0.8)
                changed, self.spawn_pos = imgui.input_float3("##SpawnPos", *self.spawn_pos)
                self.spawn_pos = list(self.spawn_pos)
                
                imgui.spacing()
                if imgui.button("Add to Scene", width=200):
                    self.add_shape_requested = True
            
            imgui.same_line()
            if imgui.button("Clear Scene", width=-1):
                self.clear_scene_requested = True
                self.spawn_pos = [0.0, 0.0, 0.0]
        
        expanded_app, _ = imgui.collapsing_header("2. APPEARANCE", flags=imgui.TREE_NODE_DEFAULT_OPEN)
        if expanded_app and not self.is_ai_mode:
            _, self.is_depth_map = imgui.checkbox("Depth Map Mode", self.is_depth_map)
            _, self.is_wireframe = imgui.checkbox("Wireframe Overlay", self.is_wireframe)
            
            if len(scene_objects) > 0 and not self.is_depth_map:
                active_obj = scene_objects[self.selected_scene_obj_idx]
                imgui.text_colored(f"Editing Material for: {active_obj.name}", 0.4, 1.0, 0.4)
                
                _, active_obj.render_mode = imgui.combo("Shading", active_obj.render_mode, ["(A) Flat Color", "(B) Vertex Color", "(C) Phong Lighting", "(D) Texture Map", "(E) Combination"])
                
                if active_obj.render_mode in [3, 4]:
                    imgui.spacing()
                    enter, active_obj.texture_filepath = imgui.input_text("Tex Image", active_obj.texture_filepath, 256, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                    if imgui.button("Apply Texture", width=-1) or enter: 
                        self.texture_changed = True
                        self.target_tex_obj_idx = self.selected_scene_obj_idx 

                if active_obj.render_mode == 0: 
                    imgui.spacing()
                    _, active_obj.flat_color = imgui.color_edit3("Base Color", *active_obj.flat_color)

                if active_obj.render_mode in [2, 4]:
                    imgui.spacing(); imgui.separator(); imgui.spacing()
                    imgui.text_colored("Global Light Sources:", 1.0, 0.8, 0.2)
                    _, self.lights[0] = imgui.checkbox("[1] Sun Light (Directional)", self.lights[0])
                    _, self.lights[1] = imgui.checkbox("[2] Warm Point Light", self.lights[1])
                    _, self.lights[2] = imgui.checkbox("[3] Cool Point Light", self.lights[2])
        
        expanded_trans, _ = imgui.collapsing_header("3. SCENE TRANSFORMS", flags=imgui.TREE_NODE_DEFAULT_OPEN)
        if expanded_trans and not self.is_ai_mode:
            if len(scene_objects) == 0:
                imgui.text_colored("Scene is empty. Add a shape first.", 0.5, 0.5, 0.5)
            else:
                imgui.text_colored("Select Object to Edit:", 0.4, 1.0, 0.4)
                obj_names = [obj.name for obj in scene_objects]
                self.selected_scene_obj_idx = max(0, min(self.selected_scene_obj_idx, len(scene_objects) - 1))
                _, self.selected_scene_obj_idx = imgui.combo("##ActiveObj", self.selected_scene_obj_idx, obj_names)
                
                active_obj = scene_objects[self.selected_scene_obj_idx]
                
                imgui.spacing()
                if imgui.button("Delete Object", width=130):
                    self.delete_obj_requested = True
                imgui.same_line()
                if imgui.button("Duplicate Object", width=-1):
                    self.duplicate_obj_requested = True
                imgui.separator()
                
                imgui.spacing()
                imgui.text_colored("[HOTKEYS] MOUSE MANIPULATION:", 1.0, 0.8, 0.0)
                imgui.text("- L-Click on Object : Select directly")
                imgui.text("- CTRL + Scroll : Scale Object")
                imgui.text("- SHIFT + L-Drag : Rotate Object")
                imgui.text("- SHIFT + R-Drag : Pan Object")
                imgui.spacing()
                
                _, active_obj.scale = imgui.slider_float("Scale", active_obj.scale, 0.1, 5.0)
                _, active_obj.rot_x = imgui.slider_float("Rotate X", active_obj.rot_x, -180.0, 180.0)
                _, active_obj.rot_y = imgui.slider_float("Rotate Y", active_obj.rot_y, -180.0, 180.0)
                _, active_obj.pos_x = imgui.slider_float("Pan X", active_obj.pos_x, -10.0, 10.0)
                _, active_obj.pos_y = imgui.slider_float("Pan Y", active_obj.pos_y, -10.0, 10.0)
                _, active_obj.pos_z = imgui.slider_float("Pan Z", active_obj.pos_z, -10.0, 10.0)
                if imgui.button("Reset Object", width=-1):
                    active_obj.scale, active_obj.rot_x, active_obj.rot_y = 1.0, 0.0, 0.0
                    active_obj.pos_x, active_obj.pos_y, active_obj.pos_z = 0.0, 0.0, 0.0

        expanded_cam, _ = imgui.collapsing_header("4. CAMERA")
        if expanded_cam:
            _, self.selected_cam_idx = imgui.combo("Active Camera", self.selected_cam_idx, ["1. Front", "2. Top-Down", "3. Free"])
            
            cam = cameras[self.selected_cam_idx]
            imgui.spacing()
            _, cam.distance = imgui.slider_float("Zoom", cam.distance, 1.0, 50.0)
            _, cam.azimuth = imgui.slider_float("Azimuth", cam.azimuth, -180.0, 180.0)
            _, cam.elevation = imgui.slider_float("Elevation", cam.elevation, -89.0, 89.0)
            
            imgui.text("Target Focus:")
            _, cam.target[0] = imgui.slider_float("Target X", cam.target[0], -10.0, 10.0)
            _, cam.target[1] = imgui.slider_float("Target Y", cam.target[1], -10.0, 10.0)
            _, cam.target[2] = imgui.slider_float("Target Z", cam.target[2], -10.0, 10.0)
            
            if imgui.button("Reset Camera", width=-1):
                cam.distance, cam.azimuth, cam.elevation = 15.0, 0.0, 20.0
                cam.target = [0.0, 0.0, 0.0]

        imgui.end()

        io = imgui.get_io()
        imgui.set_next_window_size(420, 860, imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(io.display_size.x - 440, 20, imgui.FIRST_USE_EVER)
        
        imgui.begin("AI OPTIMIZATION (PART 1.2)", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        
        
        imgui.spacing()
        changed_ai, self.is_ai_mode = imgui.checkbox(">>> ENABLE AI SIMULATION MODE <<<", self.is_ai_mode)
        _, self.is_wireframe = imgui.checkbox("Wireframe Overlay", getattr(self, 'is_wireframe', False))
        if changed_ai:
            if self.is_ai_mode:
                self.loss_changed = True
                self.reset_requested = True
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if self.is_ai_mode:
            expanded_env, _ = imgui.collapsing_header("A. Loss Landscape Setup", flags=imgui.TREE_NODE_DEFAULT_OPEN)
            if expanded_env:
                changed_loss, new_loss_idx = imgui.combo("Function", self.selected_loss_idx, self.loss_funcs)
                if changed_loss: 
                    self.selected_loss_idx = new_loss_idx
                    self.loss_changed = True
                    self.force_load_ai = False 
                
                if self.loss_funcs[self.selected_loss_idx] == "Custom":
                    imgui.spacing()
                    enter, self.custom_loss_str = imgui.input_text("Z = f(x,y)", self.custom_loss_str, 256, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                    if imgui.button("Generate Mesh", width=-1) or enter: 
                        self.loss_changed = True; self.force_load_ai = True

                imgui.spacing()
                imgui.text_colored("God Mode Controls:", 1.0, 0.8, 0.2)
                _, self.custom_z_scale = imgui.slider_float("Mountain Z-Scale", self.custom_z_scale, 0.001, 1.0, format="%.3f")
                
                changed_x, self.start_x = imgui.slider_float("Drop Pos X", self.start_x, -10.0, 10.0)
                changed_y, self.start_y = imgui.slider_float("Drop Pos Y", self.start_y, -10.0, 10.0)
                if changed_x or changed_y:
                    self.reposition_requested = True 

            expanded_ai, _ = imgui.collapsing_header("B. Agents & Hyperparams", flags=imgui.TREE_NODE_DEFAULT_OPEN)
            if expanded_ai:
                imgui.spacing()
                imgui.text_colored("Optimization Algorithms:", 0.4, 0.8, 1.0)
                _, self.opt_active[0] = imgui.checkbox("GD / SGD", self.opt_active[0])
                _, self.opt_active[1] = imgui.checkbox("Momentum", self.opt_active[1])
                _, self.opt_active[2] = imgui.checkbox("Nesterov", self.opt_active[2])
                _, self.opt_active[3] = imgui.checkbox("RMSprop", self.opt_active[3])
                _, self.opt_active[4] = imgui.checkbox("Adam", self.opt_active[4])
                
                imgui.spacing(); imgui.separator(); imgui.spacing()
                
                imgui.text_colored("Stochastic Noise Level:", 0.4, 0.8, 1.0)
                _, self.noise_level = imgui.slider_float("##Noise", getattr(self, 'noise_level', 0.0), 0.0, 2.0, format="%.2f")
                if getattr(self, 'noise_level', 0.0) == 0.0: imgui.text("Current: Batch GD (Exact)")
                elif getattr(self, 'noise_level', 0.0) < 1.0: imgui.text("Current: Mini-batch SGD")
                else: imgui.text("Current: Pure SGD (High Variance)")
                
                imgui.spacing()
                _, self.lr = imgui.slider_float("Learning Rate", self.lr, 0.0001, 0.2, format="%.4f")
                _, self.momentum_beta = imgui.slider_float("Momentum Beta", self.momentum_beta, 0.0, 0.99, format="%.2f")
                _, self.max_epochs = imgui.slider_int("Max Epochs", self.max_epochs, 100, 5000)
                _, self.steps_per_sec = imgui.slider_int("Sim Speed (FPS)", self.steps_per_sec, 1, 120)

            expanded_sim, _ = imgui.collapsing_header("C. Dashboard & Metrics", flags=imgui.TREE_NODE_DEFAULT_OPEN)
            if expanded_sim:
                imgui.spacing()
                
                if self.sim_playing:
                    if imgui.button("PAUSE", width=110): self.sim_playing = False
                else:
                    if imgui.button("PLAY", width=110): self.sim_playing = True
                imgui.same_line()
                
                if imgui.button("RESET", width=110): self.reset_requested = True
                
                imgui.spacing()
                imgui.text(f"Progress: Epoch {current_epoch} / {self.max_epochs}")
                imgui.separator()
                
                for i, opt in enumerate(optims):
                    if self.opt_active[i]:
                        imgui.text_colored(f"{opt.name} Loss: {opt.z:.4f} | Grad: {opt.grad_mag:.4f}", *opt.color)
                        imgui.text_colored(f"    Pos: ({opt.x:.2f}, {opt.y:.2f})", *opt.color)
                        imgui.spacing()

        imgui.end()