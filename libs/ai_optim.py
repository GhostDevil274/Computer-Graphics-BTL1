import numpy as np

class LossFunction:
    @staticmethod
    def get_val_and_grad(name, x, y, custom_func_str=""):
        if name == "Quadratic":
            z = x**2 + y**2
            return z, 2*x, 2*y
        elif name == "Booth":
            z = (x + 2*y - 7)**2 + (2*x + y - 5)**2
            return z, 10*x + 8*y - 34, 8*x + 10*y - 38
        elif name == "Himmelblau":
            z = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
            return z, 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7), 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
        elif name == "Rosenbrock":
            z = (1 - x)**2 + 100 * (y - x**2)**2
            return z, -2*(1 - x) - 400*x*(y - x**2), 200 * (y - x**2)
        elif name == "Custom":
            h = 1e-5
            safe_dict = {
                "x": x, "y": y,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                "abs": np.abs, "pi": np.pi, "e": np.e
            }
            try:
                z = eval(custom_func_str, {"__builtins__": None}, safe_dict)
                
                safe_dict["x"] = x + h
                z_x_plus = eval(custom_func_str, {"__builtins__": None}, safe_dict)
                safe_dict["x"] = x - h
                z_x_minus = eval(custom_func_str, {"__builtins__": None}, safe_dict)
                dx = (z_x_plus - z_x_minus) / (2 * h)
                
                safe_dict["x"] = x
                safe_dict["y"] = y + h
                z_y_plus = eval(custom_func_str, {"__builtins__": None}, safe_dict)
                safe_dict["y"] = y - h
                z_y_minus = eval(custom_func_str, {"__builtins__": None}, safe_dict)
                dy = (z_y_plus - z_y_minus) / (2 * h)
                
                return float(z), float(dx), float(dy)
            except Exception as e: return 0.0, 0.0, 0.0
        return 0, 0, 0

class Optimizer:
    def __init__(self, name, color, start_x, start_y):
        self.name, self.color = name, color
        self.x, self.y, self.z = start_x, start_y, 0.0
        self.history = []
        self.is_finished = False
        self.grad_mag = 0.0

    def reset(self, start_x, start_y):
        self.x, self.y = start_x, start_y
        self.history = []
        self.is_finished = False
        self.grad_mag = 0.0
        
    def add_noise(self, dx, dy, noise_level):
        if noise_level > 0:
            dx += np.random.normal(0, noise_level * abs(dx) + 0.1)
            dy += np.random.normal(0, noise_level * abs(dy) + 0.1)
        return dx, dy

class GradientDescent(Optimizer):
    def step(self, loss_name, lr, noise_level=0.0, custom_func_str=""):
        if self.is_finished: return
        self.z, dx, dy = LossFunction.get_val_and_grad(loss_name, self.x, self.y, custom_func_str)
        self.grad_mag = np.sqrt(dx**2 + dy**2) 
        dx, dy = np.clip(dx, -500.0, 500.0), np.clip(dy, -500.0, 500.0)
        
        dx, dy = self.add_noise(dx, dy, noise_level)
            
        if self.grad_mag < 0.01: self.is_finished = True
        self.history.append((self.x, self.y, self.z))
        self.x -= lr * dx
        self.y -= lr * dy

class Momentum(Optimizer):
    def __init__(self, name, color, start_x, start_y, beta=0.9):
        super().__init__(name, color, start_x, start_y)
        self.beta, self.vx, self.vy = beta, 0.0, 0.0
        
    def reset(self, start_x, start_y):
        super().reset(start_x, start_y)
        self.vx, self.vy = 0.0, 0.0

    def step(self, loss_name, lr, noise_level=0.0, custom_func_str=""):
        if self.is_finished: return
        self.z, dx, dy = LossFunction.get_val_and_grad(loss_name, self.x, self.y, custom_func_str)
        self.grad_mag = np.sqrt(dx**2 + dy**2) 
        dx, dy = np.clip(dx, -500.0, 500.0), np.clip(dy, -500.0, 500.0) 
        
        dx, dy = self.add_noise(dx, dy, noise_level)
        
        if self.grad_mag < 0.01: self.is_finished = True
        self.history.append((self.x, self.y, self.z))
        self.vx = self.beta * self.vx + lr * dx
        self.vy = self.beta * self.vy + lr * dy
        self.x -= self.vx
        self.y -= self.vy

class Nesterov(Optimizer):
    def __init__(self, name, color, start_x, start_y, beta=0.9):
        super().__init__(name, color, start_x, start_y)
        self.beta, self.vx, self.vy = beta, 0.0, 0.0

    def reset(self, start_x, start_y):
        super().reset(start_x, start_y)
        self.vx, self.vy = 0.0, 0.0

    def step(self, loss_name, lr, noise_level=0.0, custom_func_str=""):
        if self.is_finished: return
        self.z, dx, dy = LossFunction.get_val_and_grad(loss_name, self.x, self.y, custom_func_str)
        self.grad_mag = np.sqrt(dx**2 + dy**2)
        dx, dy = np.clip(dx, -500.0, 500.0), np.clip(dy, -500.0, 500.0)
        
        dx, dy = self.add_noise(dx, dy, noise_level)

        if self.grad_mag < 0.01: self.is_finished = True
        self.history.append((self.x, self.y, self.z))
        
        self.vx = self.beta * self.vx + lr * dx
        self.vy = self.beta * self.vy + lr * dy
        
        self.x -= (self.beta * self.vx + lr * dx)
        self.y -= (self.beta * self.vy + lr * dy)

# CLASS MỚI: RMSprop
class RMSprop(Optimizer):
    def __init__(self, name, color, start_x, start_y, decay_rate=0.9):
        super().__init__(name, color, start_x, start_y)
        self.decay_rate = decay_rate
        self.sq_grad_x, self.sq_grad_y = 0.0, 0.0

    def reset(self, start_x, start_y):
        super().reset(start_x, start_y)
        self.sq_grad_x, self.sq_grad_y = 0.0, 0.0

    def step(self, loss_name, lr, noise_level=0.0, custom_func_str=""):
        if self.is_finished: return
        self.z, dx, dy = LossFunction.get_val_and_grad(loss_name, self.x, self.y, custom_func_str)
        self.grad_mag = np.sqrt(dx**2 + dy**2)
        dx, dy = np.clip(dx, -500.0, 500.0), np.clip(dy, -500.0, 500.0)
        
        dx, dy = self.add_noise(dx, dy, noise_level)

        if self.grad_mag < 0.01: self.is_finished = True
        self.history.append((self.x, self.y, self.z))
        
        self.sq_grad_x = self.decay_rate * self.sq_grad_x + (1 - self.decay_rate) * dx**2
        self.sq_grad_y = self.decay_rate * self.sq_grad_y + (1 - self.decay_rate) * dy**2

        self.x -= (lr / (np.sqrt(self.sq_grad_x) + 1e-8)) * dx
        self.y -= (lr / (np.sqrt(self.sq_grad_y) + 1e-8)) * dy

class Adam(Optimizer):
    def __init__(self, name, color, start_x, start_y, beta1=0.9, beta2=0.999):
        super().__init__(name, color, start_x, start_y)
        self.beta1, self.beta2 = beta1, beta2
        self.m_x, self.m_y, self.v_x, self.v_y, self.t = 0.0, 0.0, 0.0, 0.0, 0
        
    def reset(self, start_x, start_y):
        super().reset(start_x, start_y)
        self.m_x, self.m_y, self.v_x, self.v_y, self.t = 0.0, 0.0, 0.0, 0.0, 0

    def step(self, loss_name, lr, noise_level=0.0, custom_func_str=""):
        if self.is_finished: return
        self.z, dx, dy = LossFunction.get_val_and_grad(loss_name, self.x, self.y, custom_func_str)
        self.grad_mag = np.sqrt(dx**2 + dy**2)
        dx, dy = np.clip(dx, -500.0, 500.0), np.clip(dy, -500.0, 500.0) 
        
        dx, dy = self.add_noise(dx, dy, noise_level)
        
        if self.grad_mag < 0.01: self.is_finished = True
        self.history.append((self.x, self.y, self.z))
        self.t += 1
        self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * dx
        self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * dy
        self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * (dx**2)
        self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * (dy**2)
        m_x_hat = self.m_x / (1 - self.beta1**self.t)
        m_y_hat = self.m_y / (1 - self.beta1**self.t)
        v_x_hat = self.v_x / (1 - self.beta2**self.t)
        v_y_hat = self.v_y / (1 - self.beta2**self.t)
        self.x -= lr * m_x_hat / (np.sqrt(v_x_hat) + 1e-8)
        self.y -= lr * m_y_hat / (np.sqrt(v_y_hat) + 1e-8)