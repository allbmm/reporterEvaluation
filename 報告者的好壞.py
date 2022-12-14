# -*- coding: utf-8 -*-

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

chek=0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic=mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) 
    
mp_drawing = mp.solutions.drawing_utils#繪圖方法

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)#繪圖參數設定

#cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture('/Users/bingjun/Downloads/690566674.058419.mp4')
cap = cv2.VideoCapture('C:/Users/to4/Desktop/111-1/hf/vidio/690566674.058419.mp4')
#cap = cv2.VideoCapture('C:/Users/to4/Desktop/111-1/hf/vidio/690566674.225508.mp4')
#cap = cv2.VideoCapture('/C:/Users/to4/Desktop/111-1/hf/vidio/690566673.976856.mp4')
#cap = cv2.VideoCapture('C:/Users/to4/Desktop/111-1/hf/vidio/690566673.898166.mp4')


turn_right = []
turn_left = []
turn_down = []
turn_up = []
turn_foward = []
no_face= []
num1=0
prnum1=0
num2=0
prnum2=0
num3=0
num4=0
num5=0
prnum5=0
no_face_number=0
#32～43為了圓餅圖的資料而設計

#臉部校正數據
face_x=5
face_y=0
face_z=0

#計數器
cal_bueaty=0#平均看所有方向

cal_noface=0
prcal_noface=0

plus_cal_atten=0
prplus_cal_atten=0

min_cal_atten=0#注視太長的次數
prmin_cal_atten=0#注視太長的次數

plus_cal_120=0#120計次數
prplus_cal_120=0

pose_center=0#以加速度的變化太大的次數，計算一個pose的量值
prpose_center=0

#紀錄器
prplus_cal_atten_2=0
prplus_cal_atten_3=0
prmin_cal_atten_2=0
prmin_cal_atten_3=0
prcal_noface_2=0
prcal_noface_3=0
prplus_cal_120_2=0
prplus_cal_120_3=0
prpose_center_2=0
prpose_center_3=0
prnum5_2=0
prnum2_2=0
prnum1_2=0
prnum5_3=0
prnum2_3=0
prnum1_3=0

width_ratio=cap.get(cv2.CAP_PROP_FRAME_WIDTH)/1280
height_ratio=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/720

cc=0#總共迴圈的次數
cc_for_haveface=0#有臉的計數

noface_time=0
pre_p1=0#前一次的鼻頭座標
center=0#肩膀中心座標center
pre_center=0#之前的中心座標
delta_center=0#中心座標的改變量
nose_distance=0#鼻頭的位移、轉動量值
nose_distance_way=0#鼻頭轉動方向
pre_nose_distance_way=0#上一次鼻頭轉動方向

attention_look=0#注視次數累積
no_atten=0
look120_scan=0#120掃視

acc_center=0#計算前後位移的變化（加速度）
pre_delta_center=0#前一次的位移量
cal_delta_center=0#計算加速度的變化次數

noface_time_start=0#沒有臉的時間開始計時
noface_time_end=0#沒臉的時間結束計時

X_angle = []
Y_angle = []
time_start = time.time() #開始計時

  
while cap.isOpened():
       
        #（success：image有東西就會回傳true 沒東西就是false） （image：一開始的那個影像格）
        success, image = cap.read()
        #imag=cap.resize(image,(0,0), fx=2,fy=2)
        cc=cc+1
        
        
        if success==False or cv2.waitKey(5) & 0xFF == 27:
            if plus_cal_atten/cc>=1/10:#注視觀眾
                atten='優'
            else:
                atten='劣'
            if min_cal_atten/cc>=1/10:#no atten
                noatten='過多'
            elif min_cal_atten==0:
                noatten='無'
            else:
                noatten='尚可'
            if cal_noface/cc>=1/10:#no face
                noface='過多'
            elif cal_noface==0:
                noface='無'
            else:
                noface='尚可'
            if  plus_cal_120>0:#120掃視
                scen='優'
            else:
                scen='無'
            if  pose_center>0:#pose
                pose='優'
            else:
                pose='無'
            if num5+num2>num1 and num2+num1>num5 and num5+num1>num2 and (num5+num2+num1)/cc>=7/10:#圓餅圖漂亮與否
                beauty = '有'
            else:
                beauty = '無'
            if num5/cc>=7/10:#是否只盯著前方的觀眾
                grade_base = '有'
            else :
                grade_base = '無'
            
            print('待改進項：')
            print('  無法偵測臉部：'+noface)#
            print('  注意力分散：'+noatten)#
            print('  幾乎注視著前方：'+grade_base)
            print('優秀項目：')
            print('  過程中平均分配視線：' +beauty)
            print('  慢速看向所有觀眾：'+scen)#
            print('  注視觀眾：'+atten)#
            print('肢體動作:'+str(pose_center))#+'/總觀察次數:'+str(int(cc/4)))#+'/總觀察次數:'+str(int(cc/4)))
            break
        if cc%100==0 :
            if cc>=300:
                prplus_cal_atten=plus_cal_atten-prplus_cal_atten
                prmin_cal_atten=min_cal_atten-prmin_cal_atten
                prcal_noface=cal_noface-prcal_noface
                prplus_cal_120=plus_cal_120-prplus_cal_120
                prpose_center=pose_center-prpose_center
                if prplus_cal_atten/300>=1/10:#注視觀眾
                    atten='優'
                else:
                    atten='劣'
                if prmin_cal_atten/300>=1/10:#no atten
                    noatten='過多'
                elif prmin_cal_atten==0:
                    noatten='無'
                else:
                    noatten='尚可'
                if prcal_noface/300>=1/10:#no face
                    noface='過多'
                elif prcal_noface==0:
                    noface='無'
                else:
                    noface='尚可'
                if  prplus_cal_120>0:#120掃視
                    scen='優'
                else:
                    scen='無'
                if prnum5+prnum2>prnum1 and prnum2+prnum1>prnum5 and prnum5+prnum1>prnum2 and (prnum5+prnum2+prnum1)/300>=7/10:#圓餅圖漂亮與否
                    beauty = '有'
                else:
                    beauty = '無'
                if prnum5/300>=7/10:#是否只盯著前方的觀眾
                    grade_base = '有'
                else :
                    grade_base = '無'
            
                print('待改進項：')
                print('  無法偵測臉部：'+noface)
                print('  注意力分散：'+noatten)
                print('  幾乎注視著前方：'+grade_base)
                print('優秀項目：')
                print('  過程中平均分配視線：' +beauty)
                print('  慢速看向所有觀眾：'+scen)
                print('  注視觀眾：'+atten)
                print('肢體動作:'+str(prpose_center))#+'/總觀察次數:'+str(int(cc/4)))#+'/總觀察次數:'+str(int(cc/4)))
                print(' ')
            
            #pr ~100
            prplus_cal_atten=prplus_cal_atten_2
            prmin_cal_atten=prmin_cal_atten_2
            prcal_noface=prcal_noface_2
            prplus_cal_120=prplus_cal_120_2
            prpose_center=prpose_center_2
            prnum5=prnum5_2
            prnum2=prnum2_2
            prnum1=prnum1_2
            #print(prpose_center)
            #100~200
            prplus_cal_atten_2=prplus_cal_atten_3
            prmin_cal_atten_2=prmin_cal_atten_3
            prcal_noface_2=prcal_noface_3
            prplus_cal_120_2=prplus_cal_120_3
            prpose_center_2=prpose_center_3
            prnum5=prnum5_3
            prnum2=prnum2_3
            prnum1=prnum1_3
            #print(prpose_center_2)
            #200~300
            prplus_cal_atten_3=plus_cal_atten
            prmin_cal_atten_3=min_cal_atten
            prcal_noface_3=cal_noface
            prplus_cal_120_3=plus_cal_120
            prpose_center_3=pose_center
            prnum5=num5
            prnum2=num2
            prnum1=num1
            #print(prpose_center_3)

        if success==True :#and cc%7==0 :

            start = time.time()
            
            #水平翻轉攝像鏡頭
            # 轉換色彩BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
            # To improve performance
            image.flags.writeable = False
            
            # Get the result
            results = face_mesh.process(image)
            results2 = holistic.process(image)#專門給pose的results
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
                
                noface_time=0
                # noface_time_end=0#把累積沒臉的時間重新計算
              
                    
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
                    
                    #120度(y=+-7)內視線的緩慢掃視
                    if -15+face_y<y<15+face_y and -5+face_x<x<5+face_x: #把臉部鎖定在120度的範圍之內(有條動,原範圍7 -7)
                        if 3>=nose_distance>0: #頭部轉動位移在1～3度內(有條動,原範圍1 3
                            if nose_distance_way!=pre_nose_distance_way and  nose_distance>=1 :#換方向+非系統跳動(>1)
                                look120_scan=0
                            else:
                                look120_scan+=1
                                if look120_scan==1:
                                    chek=y
                                if look120_scan>10:
                                    if chek-y<-1 or chek-y>1:
                                        plus_cal_120+=1
                                    else:
                                        look120_scan=0
                        else:
                            look120_scan=0
                    else:  
                        look120_scan=0
                    pre_nose_distance_way=nose_distance_way
            
                    #把頭部轉向限制在一個框框內
                    if -6+face_y<y<12+face_y and 2.5+face_x>x>-5+face_x:
                        attention_look +=1
                        if attention_look>=200: #設定一個時間，注視超過200次的迴圈次數
                            min_cal_atten+=1
                            judgement_type_addmode=5.2
                        elif 20>attention_look:           
                            no_atten=0#no atten 歸零
                            noface_time=0
                        else:
                            plus_cal_atten+=1
                        cv2.putText(image, f'attention =  {round((attention_look-4)/10,0)} times', (int(20*width_ratio),int(700*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1.5*width_ratio, (0,255,0), 2)
                    else:#沒有在取景框當中
                        no_atten+=1
                        if no_atten>20:
                            attention_look=0#重新計算注視次數，代表離開太久了
                            noface_time=0
                                
                        if no_atten>50: #如果頭部不在這個取景框中一段時間（回頭打個噴嚏之類的時間很短就不會進入這個循環，因此會繼續累積注視的次數）100數值蓋
                            judgement_type_addmode=5.2
                            min_cal_atten+=1
                            cv2.putText(image, ' inattention too long', (int(20*width_ratio),int(700*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 2*width_ratio, (0,0,255), 2)
                         
                    if y < -4+face_y:
                        text = "Looking Left"
                        num1=num1+1
                        direction_time_end=time.time()
                        turn_left.append(num1)
                         
                    elif y > 9+face_y:
                        text = "Looking Right"
                        num2=num2+1
                        direction_time_end=time.time()
                        #print(num2)
                        turn_right.append(num2)
                         
                    elif x < -3.5+face_x:
                        minues = -1
                        text = "Looking Down"
                        num3=num3+1
                        direction_time_end=time.time()
                        turn_down.append(num3)
                         
                    elif x > 6+face_x:
                        text = "Looking Up"
                        num4=num4+1
                        direction_time_end=time.time()
                        turn_up.append(num4)
                   
                    else:
                       text = "Forward"
                       num5=num5+1
                       direction_time_end=time.time()
                       turn_foward.append(num5)
                       
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
                        #cv2.putText(image, f'nose_dis= {float(nose_distance)}', (int(500*width_ratio),int(100*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 2*width_ratio, (0, 255, 0), 2)
                    
                    cv2.putText(image, text, (int(20*width_ratio),int(50*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 2*width_ratio, (0, 255, 0), 2)
                    #cv2.putText(要放文字的視窗，要放的文字，要放置的座標，字體（不用理他），字體大小，rgb的顏色，字體粗細)
                    cv2.putText(image, "x: " + str(np.round(x,2)), (int(1100*width_ratio), int(50*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1*width_ratio, (0, 0, 255), 2)
                    cv2.putText(image, "y: " + str(np.round(y,2)), (int(1100*width_ratio), int(100*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1*width_ratio, (0, 0, 255), 2)
                    cv2.putText(image, "z: " + str(np.round(z,2)), (int(1100*width_ratio), int(150*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1*width_ratio, (0, 0, 255), 2)
                                                                      #這個是調整xyz的顯示位置
                    #pose 部分：取肩膀連線的座標center為參考                                                
                    
                    if results2.pose_landmarks.landmark:
                        landmarks = results2.pose_landmarks.landmark#pose的標記
                        nose=np.array([landmarks[mp_holistic.PoseLandmark.NOSE.value].x,landmarks[mp_holistic.PoseLandmark.NOSE.value].y])
                        leftshoulder = np.array([landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y])
                        rightshoulder = np.array([landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y])
                        center=round((float(leftshoulder[0])*img_h+float(rightshoulder[0])*img_h)/2,2)

                        
                        if pre_center!=0:
                            delta_center=np.abs(center-pre_center)
                            if pre_delta_center!=0:
                                acc_center = np.abs(pre_delta_center-delta_center)
                                if acc_center > 1:
                                    cal_delta_center+=1
                                    if cal_delta_center>4:
                                        pose_center+=1
                                        
                                        cal_delta_center=0
                            cv2.putText(image, f'BODY MOVEMENTS:{(int(pose_center))}', (int(100*width_ratio),int(650*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1.5*width_ratio, (0,255,0), 2)
                                                   
                                    
                        pre_center=center
                        pre_delta_center=delta_center
                    
                end = time.time()
                totalTime = end - start
                
                
                fps = 1 / totalTime
                
                #cv2.putText(image, f'FPS: {int(fps)}', (int(1000*width_ratio),int(700*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1.5*width_ratio, (0,255,0), 2)
                pre_p1=p1
                #從noface狀態轉換到有face狀態後，計算有face維持多久，若有足夠時間，即重新計算noface_time
                if noface_time!=0 and cc_for_haveface >= 10: 
                    noface_time=0
        
            else:# no_face_number !=0: 
                #程式到這裡代表沒有偵測到臉，一開始程式卡頓的話不會跳出noface
                no_face_number = no_face_number +1 #for 圓餅圖
                
                noface_time+=1
                no_face.append(no_face_number)
                cv2.putText(image, "no face!!!" , (int(150*width_ratio), int(200*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 4, (138, 43, 226), 10)
                noface_time_end=time.time()
                if noface_time_start!=0:
                   cv2.putText(image,"time:"+str(int(noface_time_end-noface_time_start))+"s", (int(300*width_ratio), int(500*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 3*width_ratio, (138, 43, 226), 3)
                cv2.putText(image,"time:"+str(int(noface_time_end-noface_time_start))+"s", (int(300*width_ratio), int(500*height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 3*width_ratio, (138, 43, 226), 3)
           
            if noface_time>20:#累積（沒有找到臉）大於迴圈次數200
                no_atten=0
                attention_look=0    
            if noface_time>30:
                cal_noface+=1
                judgement_type_addmode=6
                cv2.putText(image,"no face too long!!", (int(100*width_ratio),int(300*height_ratio)),cv2.FONT_HERSHEY_SIMPLEX, 4*width_ratio, (138, 42, 226), 3)
            
            # if cc%10==0:
            #     # 309～319即時顯示變化的圓餅圖
            #     y = np.array([len(turn_right), len(turn_left), len(turn_up), len(turn_foward), len(turn_down),len(no_face)])
            #     #len() 括弧裡面的字元長度
            #     plt.pie(y,
            #     #plt.savefig("/Users/bingjun/Desktop/人因工程/headpose_Pie_chart"+str(cc)+".jpg") #綠色這裡要改成自己要存的地方資料夾
            #     labels=['Looking Right','Looking Left','Looking Up','Forward','Looking Down','no_face'], # 设置饼图标签
            #     colors=["#65a479", "#d5695d", "#5d8ca8", "#FF5151", "#a564c9","#FFFFBB"], # 设置饼图颜色
            #     explode=(0, 0, 0, 0, 0,0), # 第二部分突出显示，值越大，距离中心越远
            #     autopct='%.2f%%')
        
            #     plt.title("Head Pose Estimation"+str(cc%10))
            #     #plt.savefig("C:/Users/to4/Desktop/111-1/hf/data/headpose_Pie_chart"+str(cc)+".jpg") #綠色這裡要改成自己要存的地方資料夾
            #     plt.show()
            cv2.imshow('Head Pose Estimation', image)
        
            #判斷基本分
             #total=num1+num2+num3+num4+num5+no_face_number
            

XX_angle = pd.DataFrame(X_angle)
YY_angle = pd.DataFrame(Y_angle)
XY_angle = pd.DataFrame(XX_angle)
XY_angle = pd.concat([XX_angle,YY_angle],axis=1)              
XXYY_angle = pd.DataFrame(XY_angle)
#XXYY_angle.to_excel("C:/Users/to4/Desktop/111-1/hf/data/head pose estimation.xlsx")  
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
        colors=[ "#d5695d", "#5d8ca8", "#FF5151","#65a479", "#a564c9","#FFFFBB"], # 设置饼图颜色
        explode=(0, 0, 0, 0, 0,0), # 第二部分突出显示，值越大，距离中心越远
        autopct='%.2f%%')
plt.title("Final Result")
#plt.savefig('C:/Users/to4/Desktop/111-1/hf/data/headpose_Pie_chart.jpg') #綠色這裡要改成自己要存的地方資料夾
plt.show()


