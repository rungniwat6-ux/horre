import cv2
import numpy as np

def match_brightness_contrast(source, target):
    s_ycrcb = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)
    t_ycrcb = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
    
    s_y, s_cr, s_cb = cv2.split(s_ycrcb)
    t_y, t_cr, t_cb = cv2.split(t_ycrcb)
    
    s_mean, s_std = cv2.meanStdDev(s_y)
    t_mean, t_std = cv2.meanStdDev(t_y)
    s_y_mean, s_y_std = s_mean[0][0], s_std[0][0]
    t_y_mean, t_y_std = t_mean[0][0], t_std[0][0]
    
    s_y_std = max(s_y_std, 1e-5)
    
    s_y = ((s_y - s_y_mean) * (t_y_std / s_y_std)) + t_y_mean
    s_y = np.clip(s_y, 0, 255).astype(np.uint8)
    
    result = cv2.merge([s_y, s_cr, s_cb])
    return cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)

def match_color_lab(source, target):
    s_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    t_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    s_mean, s_std = cv2.meanStdDev(s_lab)
    t_mean, t_std = cv2.meanStdDev(t_lab)
    
    s_mean = s_mean.reshape((3,))
    s_std = s_std.reshape((3,))
    t_mean = t_mean.reshape((3,))
    t_std = t_std.reshape((3,))
    
    for i in range(3):
        std_ratio = t_std[i] / max(s_std[i], 1e-5)
        s_lab[:, :, i] = ((s_lab[:, :, i] - s_mean[i]) * std_ratio) + t_mean[i]
        
    s_lab = np.clip(s_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(s_lab, cv2.COLOR_LAB2BGR)
