# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 22:37:51 2022

@author: to4
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 22 19:05:09 2022

@author: TINA
"""


import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import scipy.stats as st
import math
#%%
time_start = time.time() #開始計時



mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#cap = cv2.VideoCapture('C:/Users/TINA/Desktop/smart_classroom/head pose estimation/AI model video/good exsample_4.mp4')


turn_right = []
turn_left = []
turn_down = []
turn_up = []
turn_foward = []
num1=0
num2=0
num3=0
num4=0
num5=0
X_angle = []
Y_angle = []
while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
            print(x,y)
            X_angle.append(x)
            Y_angle.append(y)
            
            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
                print("Looking Left")
                num1=num1+1
                print(num1)
                turn_left.append(num1)
                
            elif y > 10:
                text = "Looking Right"
                print("Looking Right")
                num2=num2+1
                print(num2)
                turn_right.append(num2)
                
            elif x < -10:
                text = "Looking Down"
                print("Looking Down")
                num3=num3+1
                print(num3)
                turn_down.append(num3)
            elif x > 10:
                text = "Looking Up"
                print("Looking Up")
                num4=num4+1
                print(num4)
                turn_up.append(num4)
            else:
                text = "Forward"
                print("Forward")
                num5=num5+1
                print(num5)
                turn_foward.append(num5)

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        end = time.time()
        totalTime = end - start
        
        fps = 1 / totalTime
        #print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)


    cv2.imshow('Head Pose Estimation', image)


    if cv2.waitKey(5) & 0xFF == 27:
       break
    

    
XX_angle = pd.DataFrame(X_angle)
YY_angle = pd.DataFrame(Y_angle)
XY_angle = pd.DataFrame(XX_angle)
XY_angle = pd.concat([XX_angle,YY_angle],axis=1)              
XXYY_angle = pd.DataFrame(XY_angle)
XXYY_angle.to_excel("C:/Users/TINA/Desktop/smart_classroom/head pose estimation/head pose estimation.xlsx")  







    
cap.release()
cv2.destroyAllWindows()

time_end = time.time()    #結束計時
time_c= time_end - time_start   #執行所花時間
print('time cost', time_c, 's')

y = np.array([len(turn_right), len(turn_left), len(turn_up), len(turn_foward), len(turn_down)])

plt.pie(y,
        labels=['Looking Right','Looking Left','Looking Up','Forward','Looking Down'], # 设置饼图标签
        colors=["#65a479", "#d5695d", "#5d8ca8", "#FF5151", "#a564c9"], # 设置饼图颜色
        explode=(0, 0, 0, 0, 0), # 第二部分突出显示，值越大，距离中心越远
        autopct='%.2f%%')
  
plt.title("Head Pose Estimation")
plt.savefig('C:/Users/TINA/Desktop/smart_classroom/head pose estimation/diagram/headpose_Pie_chart.jpg')
plt.show()



    
cap.release()
cv2.destroyAllWindows()

time_end = time.time()    #結束計時
time_c= time_end - time_start   #執行所花時間
print('time cost', time_c, 's')

y = np.array([len(turn_right), len(turn_left), len(turn_up), len(turn_foward), len(turn_down)])

plt.pie(y,
        labels=['Looking Right','Looking Left','Looking Up','Forward','Looking Down'], # 设置饼图标签
        colors=["#65a479", "#d5695d", "#5d8ca8", "#FF5151", "#a564c9"], # 设置饼图颜色
        explode=(0, 0, 0, 0, 0), # 第二部分突出显示，值越大，距离中心越远
        autopct='%.2f%%')
  
plt.title("Head Pose Estimation")
plt.savefig('C:/Users/TINA/Desktop/smart_classroom/head pose estimation/diagram/headpose_Pie_chart.jpg')
plt.show()