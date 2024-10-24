import os
import os.path as osp
from typing import Tuple
from tqdm import tqdm
import cv2
import numpy as np

from utils.dataclasses import Center


def draw_frame(img_or_path,
               center: Center,
               color: Tuple,
               radius : int = 5,
               thickness : int = -1,
    ):
        if osp.isfile(img_or_path):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path

        xy   = center.xy
        visi = center.is_visible
        if visi:
            x, y = xy
            x, y = int(x), int(y)
            img  = cv2.circle(img, (x,y), radius, color, thickness=thickness)
        
        return img


def draw_frame_multiball(img_or_path,
               xs: list,
               ys: list,
               is_visible: bool,
               color: Tuple,
               radius: int = 5,
               thickness: int = -1,
               ):
    if os.path.isfile(img_or_path):
        img = cv2.imread(img_or_path)
    else:
        img = img_or_path

    if is_visible:
        for x, y in zip(xs, ys):
            x, y = int(x), int(y)
            img = cv2.circle(img, (x, y), radius, color, thickness=thickness)

    return img

        
    # if is_visible:
    #     # Prepare a matrix for circle detection
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.medianBlur(gray, 5)

    #     find_length=50
        
    #     # Iterate through each point
    #     for x, y in zip(xs, ys):
    #         x, y = int(x), int(y)
    #         # Define the region of interest (ROI)
    #         x_min = max(x - find_length, 0)
    #         y_min = max(y - find_length, 0)
    #         x_max = min(x + find_length, gray.shape[1])
    #         y_max = min(y + find_length, gray.shape[0])
            
    #         roi_gray = gray[y_min:y_max, x_min:x_max]

        
    #         # HoughCircles to detect circles in the ROI
    #         circles = cv2.HoughCircles(roi_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
    #                                    param1=100, param2=50, minRadius=15, maxRadius=40)
            

    #         draw_both = False
            
    #         if circles is not None:
    #             circles = np.round(circles[0, :]).astype("int")
    #             for (xc, yc, rc) in circles:
    #                 # Adjust circle coordinates relative to the original image
    #                 xc += x_min
    #                 yc += y_min
    #                 if np.sqrt((x - xc) ** 2 + (y - yc) ** 2) < radius * 2:
    #                     draw_both = True
    #                     # Draw detected circle
    #                     img = cv2.circle(img, (xc, yc), rc, (0, 255, 0), thickness=2)
    #                     break
            
    #         if draw_both:
    #             # Draw the input point
    #             img = cv2.circle(img, (x, y), radius, color, thickness=thickness)
                
    # return img



def gen_video(video_path, 
              vis_dir, 
              resize=1.0, 
              fps=30.0, 
              fourcc='mp4v'
):

    fnames = os.listdir(vis_dir)
    # fnames.sort()
    fnames = sorted(fnames, key=lambda x: int(x.split('.')[0]))
    h,w,_   = cv2.imread(osp.join(vis_dir, fnames[0])).shape
    im_size = (int(w*resize), int(h*resize))
    fourcc  = cv2.VideoWriter_fourcc(*fourcc)
    out     = cv2.VideoWriter(video_path, fourcc, fps, im_size)

    for fname in tqdm(fnames):
        im_path = osp.join(vis_dir, fname)
        im      = cv2.imread(im_path)
        im = cv2.resize(im, None, fx=resize, fy=resize)
        out.write(im)

