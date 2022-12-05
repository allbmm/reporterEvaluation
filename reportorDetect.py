# -*- coding: utf-8 -*-
"""
Created on Sun May 22 19:05:09 2022

@author: TINA
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 20:35:20 2022

@author: to4
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
from PIL import Image ,ImageDraw,ImageFont
#%%
time_start = time.time() #開始計時
chek=0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils#繪圖方法

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)#繪圖參數設定

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('/Users/bingjun/Downloads/廉政不好.mp4')
#cap = cv2.VideoCapture('C:/Users/to4/Desktop/111-1/hf/vidio/IMG_9422.mp4')
#cap = cv2.VideoCapture('/C:/Users/to4/Desktop/111-1/hf/vidio/690566673.976856.mp4')
#cap = cv2.VideoCapture('C:/Users/to4/Desktop/111-1/hf/vidio/1.mp4')

turn_right = []
turn_left = []
turn_down = []
turn_up = []
turn_foward = []
no_face= []
num1=0
num2=0
num3=0
num4=0
num5=0
no_face_number=0
#32～43為了圓餅圖的資料而設計

#臉部校正數據
face_x=8
face_y=2
face_z=0

numl=0
numr=0
numm=0
#計數器
cal_noface=0
cal_long_way=0
cal_longleft=0
cal_longright=0
cal_longup=0
cal_longdown=0
cal_longforward=0
cal_120add=0
cal_attenlook=0

width_ratio=cap.get(cv2.CAP_PROP_FRAME_WIDTH)/1280
height_ratio=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/720

cc=0#總共迴圈的次數
cc_for_haveface=0#有臉的計數
no_atten=0
grade_all=0#所有總分，加扣分和基本分加起來
grade_base=0#基本分數

min_for_noface=0#單獨計算noface的扣分分數
min_for_atten=0#注視太長的扣分分數
plus_for120=0#120加分
plus_cal_attenlook=0 #注視恰當(注視觀眾)的加分分數

noface_time=0
pre_p1=0#前一次的鼻頭座標
nose_distance=0#鼻頭的位移、轉動量值
nose_distance_way=0#鼻頭轉動方向
pre_nose_distance_way=0#上一次鼻頭轉動方向

delta_min=0#數有多少次的緩慢移動頭部
look120_loop=0#到目前為止，delta_min的比例
attention_look=0#注視次數累積
look120_scan=0#120掃視

judgement_type=0#0:什麼都沒有；12345:臉部朝向一個方向的累積
pre_judgement_type_addmode=0#之前的判斷
judgement_type_addmode=0#判斷是否為同個加分模式

direction_time_start=0#臉部朝向的累積開始時間
direction_time_end=0#臉部朝向的累積結束時間
direction_time=0

noface_time_start=0#沒有臉的時間開始計時
noface_time_end=0#沒臉的時間結束計時
last_text=""#最後圓餅圖顯示的文字

X_angle = []
Y_angle = []
while cap.isOpened():
    #（success：image有東西就會回傳true 沒東西就是false） （image：一開始的那個影像格）
    success, image = cap.read()
    #imag=cap.resize(image,(0,0), fx=2,fy=2)
    cc=cc+1
    
    if success==False or cv2.waitKey(5) & 0xFF == 27:
        print('扣分變動：'+str(min_for_atten+min_for_noface))
        print('  noface扣分：'+str(cal_noface)+'次/共'+str(min_for_noface)+'分')
        print('  long time one way 扣分：'+str(cal_long_way)+'次/n共'+str(min_for_atten)+'分')
        print('    Left：'+str(cal_longleft)+'次/共'+str(min_for_atten)+'分')
        print('    Right：'+str(cal_longright)+'次/共'+str(min_for_atten)+'分')
        print('    Up：'+str(cal_longup)+'次/共'+str(min_for_atten)+'分')
        print('    Down：'+str(cal_longdown)+'次/共'+str(min_for_atten)+'分')
        print('    Forward：'+str(cal_longforward)+'次/共'+str(min_for_atten)+'分')
        print('加分變動：'+str(plus_cal_attenlook+plus_for120))
        print('  120掃視加分：'+str(cal_120add)+'次/共'+str(plus_for120)+'分')
        print('  注視觀眾加分：'+str(cal_attenlook)+'次/共'+str(plus_cal_attenlook)+'分')
        print('基礎分：'+str(grade_base))
        print('總分：'+str(grade_all))
        #print(judgement_type_addmode)
       
        
        break
    
    if pre_judgement_type_addmode!=judgement_type_addmode:
        time_here_s=0
        time_here_min =0
        time_here_hr=0
        
        time_here=direction_time_start-time_start
        time_here = round(time_here, 2)
        time_here_s=time_here%60
        time_here_min = (time_here-time_here_s/60)%60
        time_here_min=int(time_here_min)
        time_here_hr=time_here_min/60
        time_here_hr=int(time_here_hr)
        
        
        if judgement_type_addmode==7:
            cal_120add+=1
            print("120掃視加分："+str(time_here_hr)+"hr"+str(time_here_min)+"min"+str(time_here_s)+"s")
            plus_for120+=1
        elif judgement_type_addmode==5.2:
            cal_long_way+=1
            if judgement_type == 1:
                cal_longleft += 1
                print("long time left："+str(time_here_hr)+"hr"+str(time_here_min)+"min"+str(time_here_s)+"s")
            elif judgement_type == 2:
                cal_longright += 1
                print("long time Right："+str(time_here_hr)+"hr"+str(time_here_min)+"min"+str(time_here_s)+"s")
            elif judgement_type == 3:
                cal_longup += 1
                print("long time Up："+str(time_here_hr)+"hr"+str(time_here_min)+"min"+str(time_here_s)+"s")
            elif judgement_type == 4:
                cal_longdown += 1
                print("long time Down："+str(time_here_hr)+"hr"+str(time_here_min)+"min"+str(time_here_s)+"s")
            elif judgement_type == 5:
                cal_longforward += 1
                print("long time Forward："+str(time_here_hr)+"hr"+str(time_here_min)+"min"+str(time_here_s)+"s")
        elif judgement_type_addmode==6:
            cal_noface += 1
            print("no face："+str(time_here_hr)+"hr"+str(time_here_min)+"min"+str(time_here_s)+"s")
        elif judgement_type_addmode==8:
            cal_attenlook+=1
            print("注視觀眾："+str(time_here_hr)+"hr"+str(time_here_min)+"min"+str(time_here_s)+"s")
        pre_judgement_type_addmode=judgement_type_addmode
    judgement_type_addmode=0
        
    start = time.time()

    #水平翻轉攝像鏡頭
    # 轉換色彩BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
      # To improve performance
    image.flags.writeable = True
    
    #再次轉換爲bgr 以和open cv 做運用
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
  
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        noface_time_start=time.time() 
        cc_for_haveface+=1
        
        noface_time_end=0#把累積沒臉的時間重新計算
      
            
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                #idx=1,xyz=?;idx=2,xyz=?
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000 )
                   
                        

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

            yp=y
            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
            #print(x,y)
            X_angle.append(x)
            Y_angle.append(y)
            #120度(y=+-7)內視線的緩慢掃視：未試出
            #y 轉越左邊數值越負，越右邊越正，中間是0
            
            if -15+face_y<y<15+face_y : #把臉部鎖定在120度的範圍之內(有條動,原範圍7 -7)and -5+face_x<x<5+face_x
            
                
                if 3>=nose_distance>0: #頭部轉動位移在1～3度內(有條動,原範圍1 3
                    if nose_distance_way!=pre_nose_distance_way and  nose_distance>=0.5 :#換方向+非系統跳動(>1)
                        look120_scan=0
                        
                    else:
                        look120_scan+=1
                        if look120_scan==1:
                            chek=y
                            print('chek'+str(chek))
                        if look120_scan>10:
                            if chek-y<-1 or chek-y>1:
                                judgement_type_addmode=7
                                print(  '')
                            else:
                                look120_scan=0
                            
                    # else:
                    #     look120_scan=0
                        
                    delta_min += 1
                    look120_loop=round(delta_min/cc,2)
                    #print('120度的範圍次數：'+str(delta_min))
                else:
                    look120_scan=0
            else:  
                look120_scan=0
            # else:
            #     look120_scan=0
            cv2.putText(image, f'120/all loop= {look120_loop}', (int(20*width_ratio),int(450*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1.5*width_ratio, (0,255,0), 2)
            print(str(look120_scan)+"/"+str(y)+"/"+str(nose_distance)+"/"+str(nose_distance_way))
            pre_nose_distance_way=nose_distance_way
            #把頭部轉向限制在一個框框內
        
            if -5+face_y<y<5+face_y and 10+face_x>x>1+face_x:
                attention_look +=1
                no_atten=0#no atten 歸零
                if attention_look>=5 : #設定一個時間，注視超過5次的迴圈次數
                    cv2.putText(image, f'attention =  {round((attention_look-4)/10,0)} times', (int(20*width_ratio),int(350*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1.5*width_ratio, (0,255,0), 2)
                    
            else:
                no_atten+=1
                if no_atten>100: #如果頭部不在這個取景框中一段時間（回頭打個噴嚏之類的時間很短就不會進入這個循環，因此會繼續累積注視的次數）
                   attention_look=0#重新計算注視次數，代表離開太久了
                   cv2.putText(image, 'not attention too long', (int(20*width_ratio),int(350*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 2*width_ratio, (0,0,255), 2)
            
       
            # See where the user's head tilting
            if y < -6+face_y:
                text = "Looking Left"
                #print("Looking Left")
                num1=num1+1
                direction_time_end=time.time()
                #print(num1)
                turn_left.append(num1)
                
                if judgement_type!=1:#如果之前的面部朝向不是向左這個類型就重新開始累積向前的時間
                    direction_time_start=time.time()#開始進行朝向左的時間計時
                    direction_time_end=0#把之前不管是怎樣面部朝向的累積時間清零
                    judgement_type=1
                 
                 
            elif y > 6+face_y:
                text = "Looking Right"
                #print("Looking Right")
                num2=num2+1
                direction_time_end=time.time()
                #print(num2)
                turn_right.append(num2)
                
                if judgement_type!=2:#如果之前的面部朝向不是向右這個類型就重新開始累積向前的時間
                    direction_time_start=time.time()#開始進行朝向右的時間計時
                    direction_time_end=0#把之前不管是怎樣面部朝向的累積時間清零
                    judgement_type=2
                 
                 
            elif x < -5+face_x:
                minues = -1
                text = "Looking Down"
                #print("Looking Down")
                num3=num3+1
                direction_time_end=time.time()
                #grade = grade + minues
                #print(num3)
                turn_down.append(num3)
                if judgement_type!=3:#如果之前的面部朝向不是向下這個類型就重新開始累積向前的時間
                    direction_time_start=time.time()#開始進行朝向下的時間計時
                    direction_time_end=0#把之前不管是怎樣面部朝向的累積時間清零
                    judgement_type=3
                 
                 
            elif x > 5+face_x:
                text = "Looking Up"
               # print("Looking Up")
                num4=num4+1
                direction_time_end=time.time()
                #print(num4)
                turn_up.append(num4)
                
                if judgement_type!=4:#如果之前的面部朝向不是向上這個類型就重新開始累積向上的時間
                    direction_time_start=time.time()#開始進行朝向上的時間計時
                    direction_time_end=0#把之前不管是怎樣面部朝向的累積時間清零
                    judgement_type=4
                 
                 
            else:
               #plus = 10 #看前面可以累積的分數 
               text = "Forward"
               #print("Forward")
               num5=num5+1
               direction_time_end=time.time()
               #print(num5)
               turn_foward.append(num5)
               
               if judgement_type!=5:#如果之前的面部朝向不是向前這個類型就重新開始累積向前的時間
                   direction_time_start=time.time()#開始進行朝向前面的時間計時
                   direction_time_end=0#把之前不管是怎樣面部朝向的累積時間清零
                   judgement_type=5
               
               
            if judgement_type==5 and direction_time_end-direction_time_start > 2 and direction_time_end-direction_time_start < 10:
                   
                #time.sleep(0.5)
                judgement_type_addmode=8
                plus_cal_attenlook = plus_cal_attenlook + 0.005
                plus_cal_attenlook = round(plus_cal_attenlook, 3)
                direction_time = round(direction_time_end-direction_time_start, 0)
                cv2.putText(image, "grade plus time : "+str(direction_time)+"s", (int(20*width_ratio),int(650*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 2*width_ratio, (255, 215, 0), 2)
               #cv2.putText(image,str(lei)+"    "+str(int(leiji_start)), (400,600),cv2.FONT_HERSHEY_SIMPLEX, 1, (138, 42, 226), 2)
              
               
                
                
            if direction_time_end-direction_time_start > 10:#如果朝向一個方向的時間>10秒
                  judgement_type_addmode=5.2
                  min_for_atten= min_for_atten - 0.005  
                  min_for_atten = round(min_for_atten, 3)
                  direction_time = round((direction_time_end-direction_time_start),2)#將臉部朝向同方向的時間取到小數點第二位
                  cv2.putText(image, "the same direction time : "+str(direction_time)+"s", (int(20*width_ratio), int(650*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 2*width_ratio, (255, 215, 0), 2)
               
            
            
          
            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10  ) , int(nose_2d[1]- x * 10))
            cv2.line(image, p1, p2, (255, 255,0), 3)
            #p1=float(''.join(map(str, p1)))/1000
            p1= float(nose_2d[1])
            
            
            #cv2.line（要放上去的地方，起始座標，結束座標，（藍，綠，紅），線條寬度）
            if pre_p1!=0:
                nose_distance=round(np.abs(pre_p1-p1),2)
                if pre_p1-p1<0:
                    nose_distance_way=1
                elif pre_p1-p1>0:
                    nose_distance_way=2
                cv2.putText(image, f'nose_dis= {float(nose_distance)}', (int(500*width_ratio),int(100*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 2*width_ratio, (0, 255, 0), 2)
                if nose_distance>=20: cv2.putText(image, 'Good! Walk around!', (int(400*width_ratio),int(50*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 2*width_ratio, (0, 255, 0), 2)
            # Add the text on the image
            #cv2ImgAddText(image, "向上看", 200, 200, (255, 255, 0), 20)
            
            cv2.putText(image, text, (int(20*width_ratio),int(50*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 2*width_ratio, (0, 255, 0), 2)
            #cv2.putText(image, str(grade/100), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 215, 0), 2)
            #cv2.putText(要放文字的視窗，要放的文字，要放置的座標，字體（不用理他），字體大小，rgb的顏色，字體粗細)
            cv2.putText(image, "x: " + str(np.round(x,2)), (int(1100*width_ratio), int(50*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1*width_ratio, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (int(1100*width_ratio), int(100*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1*width_ratio, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (int(1100*width_ratio), int(150*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1*width_ratio, (0, 0, 255), 2)
           
                                                          #這個是調整xyz的顯示位置

        end = time.time()
        totalTime = end - start
        
        
        fps = 1 / totalTime
        
        plus_point=round(plus_cal_attenlook+plus_for120,3)
        grade_variable=min_for_noface+min_for_atten+plus_cal_attenlook+plus_for120
        #print("FPS: ", fps)
        #cv2.putText(image, f'basic point: {int(grade_base)}', (20,500), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.putText(image, f'minus point: {float(min_for_noface+min_for_atten)}', (int(20*width_ratio),int(550*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1.5*width_ratio, (0,255,0), 2)
        cv2.putText(image, f'plus point: {float(plus_point)}', (int(20*width_ratio),int(600*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1.5*width_ratio, (0,255,0), 2)
        cv2.putText(image, f'base grade: {float(grade_base)}', (int(20*width_ratio),int(500*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1.5*width_ratio, (0,255,0), 2)
        cv2.putText(image, f'FPS: {int(fps)}', (int(1000*width_ratio),int(700*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1.5*width_ratio, (0,255,0), 2)
        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
        pre_p1=p1
        #從noface狀態轉換到有face狀態後，計算有face維持多久，若有足夠時間，即重新計算noface_time
        if noface_time!=0 and cc_for_haveface >= 100: 
            noface_time=0
        # if lei> 3:#如果累積（沒有找到臉）大於2秒那麼就會扣分
        #     time.sleep(0.2)
        #     plus = -1
        #     grade = grade+plus
        #     cv2.putText(image,"kou fen!!", (400,600),cv2.FONT_HERSHEY_SIMPLEX, 3, (138, 42, 226), 3)
        

    else:
        #程式到這裡代表沒有偵測到臉
        judgement_type=0
        grade_variable=min_for_noface+min_for_atten+plus_cal_attenlook+plus_for120
        no_face_number = no_face_number +1 
        noface_time+=1
        #print("no face")
        #print(no_face_number)
        no_face.append(no_face_number)
        cv2.putText(image, "no face!!!" , (int(300*width_ratio), int(100*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 4, (138, 43, 226), 10)
        #cv2.putText(image, "plus = 0" , (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (138, 43, 226), 10)
        #cv2.putText(image, "Grade = " + str(grade/100) , (300, 350), cv2.FONT_HERSHEY_SIMPLEX, 3, (138, 43, 226), 10)
        noface_time_end=time.time()
        if noface_time_start!=0:
           cv2.putText(image,"time:"+str(int(noface_time_end-noface_time_start))+"s", (int(300*width_ratio), int(500*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 3*width_ratio, (138, 43, 226), 3)
        cv2.putText(image,"time:"+str(int(noface_time_end-noface_time_start))+"s", (int(300*width_ratio), int(500*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 3*width_ratio, (138, 43, 226), 3)
   
        if noface_time>200:#累積（沒有找到臉）大於迴圈次數200
            cv2.putText(image,"no face too long!!", (int(200*width_ratio),int(300*height_ratio)),cv2.FONT_HERSHEY_SIMPLEX, 4*width_ratio, (138, 42, 226), 3)
    
    if noface_time_end - noface_time_start> 3:#如果累積（沒有找到臉）大於3秒那麼就會扣分
        #time.sleep(0.2)
        judgement_type_addmode=6
        plus = -0.01
        min_for_noface = min_for_noface + plus
        min_for_noface = round(min_for_noface, 3)
        #cv2.putText(image,leiji+"s  "+ str(grade/100), (300, 600), cv2.FONT_HERSHEY_SIMPLEX, 3, (138, 43, 226), 3)
    
        cv2.putText(image,"Deduct points!!", (int(200*width_ratio),int(300*height_ratio)),cv2.FONT_HERSHEY_SIMPLEX, 4*width_ratio, (138, 42, 226), 3)
    
    if cc%20==0:
        # 309～319即時顯示變化的圓餅圖
        y = np.array([len(turn_right), len(turn_left), len(turn_up), len(turn_foward), len(turn_down),len(no_face)])
        #len() 括弧裡面的字元長度
        plt.pie(y,
        #plt.savefig("/Users/bingjun/Desktop/人因工程/headpose_Pie_chart"+str(cc)+".jpg") #綠色這裡要改成自己要存的地方資料夾
        labels=['Looking Right','Looking Left','Looking Up','Forward','Looking Down','no_face'], # 设置饼图标签
        colors=["#65a479", "#d5695d", "#5d8ca8", "#FF5151", "#a564c9","#FFFFBB"], # 设置饼图颜色
        explode=(0, 0, 0, 0, 0,0), # 第二部分突出显示，值越大，距离中心越远
        autopct='%.2f%%')

        plt.title("Head Pose Estimation"+str(cc))
        plt.savefig("C:/Users/to4/Desktop/111-1/hf/data/headpose_Pie_chart"+str(cc)+".jpg") #綠色這裡要改成自己要存的地方資料夾
        plt.show()
    cv2.imshow('Head Pose Estimation', image)

    #判斷基本分
     #total=num1+num2+num3+num4+num5+no_face_number
    if num5/cc>=7/10:
        grade_base = 80
    elif num1/cc>=3/10 or num2/cc>=3/10 or num3/cc>=3/10 or num4/cc>=3/10 or no_face_number/cc>=3/10:
        grade_base = 40
    else:
       grade_base = 60
   
    grade_all=grade_variable+grade_base#算總分
    grade_all=round(grade_all,3)
    if grade_all>60:
        last_text="Good Presenter!"
    else:
        last_text="Bad Presenter!"
XX_angle = pd.DataFrame(X_angle)
YY_angle = pd.DataFrame(Y_angle)
XY_angle = pd.DataFrame(XX_angle)
XY_angle = pd.concat([XX_angle,YY_angle],axis=1)              
XXYY_angle = pd.DataFrame(XY_angle)
XXYY_angle.to_excel("C:/Users/to4/Desktop/111-1/hf/data/head pose estimation.xlsx")  
#綠色這裡要改成自己要存的地方資料夾


    


time_end = time.time()    #結束計時
time_c= time_end - time_start   #執行所花時間
print('time cost', time_c, 's')


cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)#window不理
y = np.array([len(turn_right), len(turn_left), len(turn_up), len(turn_foward), len(turn_down),len(no_face)])
    #len() 括弧裡面的字元長度
plt.pie(y,
        labels=['Looking Right','Looking Left','Looking Up','Forward','Looking Down','no_face'], # 设置饼图标签
        colors=["#65a479", "#d5695d", "#5d8ca8", "#FF5151", "#a564c9","#FFFFBB"], # 设置饼图颜色
        explode=(0, 0, 0, 0, 0,0), # 第二部分突出显示，值越大，距离中心越远
        autopct='%.2f%%')
plt.title("Final Result"+"\namong of grade ="+str(grade_all)+" \n\n"+last_text)
#plt.title("among of grade ="+str(grade_all),loc="center")
plt.savefig('C:/Users/to4/Desktop/111-1/hf/data/headpose_Pie_chart.jpg') #綠色這裡要改成自己要存的地方資料夾
plt.show()



