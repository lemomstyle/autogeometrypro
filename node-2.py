import torch
import numpy as np
import cv2

class ArchitectureRealit:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                # 1. 畸变搜索范围：广角建议 1.5 ~ 2.5
                "distortion_limit": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1}),
                
                # 2. 角度阈值：过滤掉非垂直的装饰线
                "max_angle_limit": ("FLOAT", {"default": 45.0, "min": 10.0, "max": 60.0}),
                
                # 3. 纵横比修正 (关键更新)：
                # 1.0 = 保持数学计算结果（可能偏胖）
                # > 1.0 = 拉长高度（变瘦/变高），通常 1.2-1.5 能恢复正常观感
                # < 1.0 = 压扁高度
                "aspect_ratio": ("FLOAT", {"default": 1.25, "min": 0.5, "max": 2.0, "step": 0.05}),
                
                # 4. 最终混合强度
                "final_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                "fill_color_hex": ("STRING", {"default": "#FFFFFF"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("corrected_image", "mask")
    FUNCTION = "correct_architecture"
    CATEGORY = "Image Processing/Geometry"

    def correct_architecture(self, image, distortion_limit, max_angle_limit, aspect_ratio, final_strength, fill_color_hex):
        results_img = []
        results_mask = []
        
        bg_color = self._hex_to_bgr(fill_color_hex)

        for tensor_img in image:
            h, w = tensor_img.shape[:2]
            img_np = (tensor_img.cpu().numpy() * 255.0).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            try:
                corrected_bgr, mask = self._solve_optimal_perspective(img_bgr, bg_color, distortion_limit, max_angle_limit, aspect_ratio, final_strength)
            except Exception as e:
                print(f"ArchitectureRealit Error: {e}")
                corrected_bgr = img_bgr
                mask = np.zeros((h, w), dtype=np.uint8)

            img_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
            img_out = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
            mask_out = torch.from_numpy(mask.astype(np.float32) / 255.0)

            results_img.append(img_out)
            results_mask.append(mask_out)

        return (torch.stack(results_img), torch.stack(results_mask))

    def _hex_to_bgr(self, hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6: return (255, 255, 255)
        return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

    def _solve_optimal_perspective(self, img, bg_color, dist_limit, max_angle, aspect_scale, strength):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- 1. LSD 线段检测 ---
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(gray)[0]
        
        if lines is None: return img, np.zeros((h, w), dtype=np.uint8)

        # --- 2. 筛选有效垂直线 ---
        valid_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length < h * 0.05: continue 

            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if angle > 90: angle -= 180
            if angle < -90: angle += 180
            
            deviation = abs(abs(angle) - 90)
            if deviation < max_angle:
                valid_lines.append([x1, y1, x2, y2, length])

        if not valid_lines: return img, np.zeros((h, w), dtype=np.uint8)
        lines_data = np.array(valid_lines)

        # --- 3. 透视求解器 (寻找最佳拉伸值 p) ---
        best_p = 0.0
        min_loss = float('inf')
        
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # 粗搜索
        search_range = np.arange(-0.2, dist_limit, 0.2)
        for p in search_range:
            shift = w * p * 0.5
            dst_pts = np.float32([[0 - shift, 0], [w + shift, 0], [0, h], [w, h]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            pts1 = lines_data[:, 0:2].reshape(-1, 1, 2).astype(np.float32)
            pts2 = lines_data[:, 2:4].reshape(-1, 1, 2).astype(np.float32)
            lengths = lines_data[:, 4]
            t_pts1 = cv2.perspectiveTransform(pts1, M)
            t_pts2 = cv2.perspectiveTransform(pts2, M)
            
            dx = np.abs(t_pts1[:, 0, 0] - t_pts2[:, 0, 0])
            loss = np.sum(dx * (lengths ** 2))
            
            if loss < min_loss:
                min_loss = loss
                best_p = p
                
        # 精细搜索
        fine_range = np.arange(best_p - 0.2, best_p + 0.2, 0.02)
        for p in fine_range:
            shift = w * p * 0.5
            dst_pts = np.float32([[0-shift, 0], [w+shift, 0], [0, h], [w, h]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            pts1 = lines_data[:, 0:2].reshape(-1, 1, 2).astype(np.float32)
            pts2 = lines_data[:, 2:4].reshape(-1, 1, 2).astype(np.float32)
            lengths = lines_data[:, 4]
            t_pts1 = cv2.perspectiveTransform(pts1, M)
            t_pts2 = cv2.perspectiveTransform(pts2, M)
            
            dx = np.abs(t_pts1[:, 0, 0] - t_pts2[:, 0, 0])
            loss = np.sum(dx * (lengths ** 2))
            
            if loss < min_loss:
                min_loss = loss
                best_p = p

        final_p = best_p * strength
        
        # --- 4. 构建透视矩阵 ---
        shift = w * final_p * 0.5
        dst_pts = np.float32([
            [0 - shift, 0], 
            [w + shift, 0], 
            [0, h], 
            [w, h]
        ])
        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # --- 5. 旋转校正 (自动回正) ---
        pts1 = lines_data[:, 0:2].reshape(-1, 1, 2).astype(np.float32)
        pts2 = lines_data[:, 2:4].reshape(-1, 1, 2).astype(np.float32)
        t_pts1 = cv2.perspectiveTransform(pts1, M_persp)
        t_pts2 = cv2.perspectiveTransform(pts2, M_persp)
        
        dxs = t_pts1[:, 0, 0] - t_pts2[:, 0, 0]
        dys = t_pts1[:, 0, 1] - t_pts2[:, 0, 1]
        angles = np.degrees(np.arctan2(dys, dxs))
        
        deviations = []
        for i, ang in enumerate(angles):
            diff = ang - 90
            if diff < -180: diff += 360
            if abs(diff) > 45: diff = ang - (-90)
            
            if lines_data[i, 4] > h * 0.1: 
                deviations.append(diff)
        
        rot_angle = 0
        if deviations:
            rot_angle = np.median(deviations)
        if abs(rot_angle) > 15: rot_angle = 0 
        
        center = (w/2, h/2)
        R = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
        M_rot = np.eye(3)
        M_rot[:2] = R
        
        M_current = np.dot(M_rot, M_persp)

        # --- 6. 纵横比修正 (Aspect Ratio Correction) ---
        # 你的需求核心在这里。
        # 如果 aspect_scale > 1.0，我们对 Y 轴进行拉伸。
        # 构造一个缩放矩阵
        M_scale = np.eye(3)
        M_scale[1, 1] = aspect_scale # Y轴缩放
        
        # 将缩放矩阵应用到当前的变换矩阵上
        M_final = np.dot(M_scale, M_current)

        # --- 7. 画布扩展与渲染 ---
        corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]]).reshape(-1, 1, 2)
        new_corners = cv2.perspectiveTransform(corners, M_final)
        
        x_min = min(new_corners[:, 0, 0])
        x_max = max(new_corners[:, 0, 0])
        y_min = min(new_corners[:, 0, 1])
        y_max = max(new_corners[:, 0, 1])
        
        new_w = int(np.ceil(x_max - x_min))
        new_h = int(np.ceil(y_max - y_min))
        
        T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        M_render = np.dot(T, M_final)
        
        warped = cv2.warpPerspective(img, M_render, (new_w, new_h), 
                                     flags=cv2.INTER_LANCZOS4, 
                                     borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=bg_color)
        
        white = np.ones((h, w), dtype=np.uint8) * 255
        mask_warp = cv2.warpPerspective(white, M_render, (new_w, new_h), 
                                        flags=cv2.INTER_NEAREST, 
                                        borderValue=0)

        return warped, mask_warp