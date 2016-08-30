import cv2
import numpy as np
import sys   

from matplotlib import pyplot as plt
sys.setrecursionlimit(10000)

#3_8.png判断有错
src = cv2.imread("4_1.png")
#调整图像
y,x,channel = src.shape
if x >= y :
    img = cv2.resize\
          (src,(700, np.int32(700.0/x*y)),interpolation=cv2.INTER_CUBIC)
    
else:
    img = cv2.resize\
          (src,(np.int32(700.0/y*x), 700),interpolation=cv2.INTER_CUBIC)


    

cv2.imshow("img",img)
cv2.waitKey(0)
y,x,channel = img.shape

 
area = x * y


b = np.zeros((y,x),np.uint8)
g = np.zeros((y,x),np.uint8)
r = np.zeros((y,x),np.uint8)
#用白平衡法解决偏色问题(不好)

b = img[:,:,0].copy()


g = img[:,:,1].copy()
r = img[:,:,2].copy()

#bgr转ycgcr
A = np.array([[65.481, 128.553, 24.966],[-81.085, 112, -30.915],[112, -93.768, -18.214]])
B = np.array([[16],[128],[128]])
C = np.repeat(B,x,1)
             
ycgcr = np.zeros(img.shape,np.uint8)

for j in range(y):
             
             n = C + 1.0/256 * np.dot(A,[r[j],g[j],b[j]])
             ycgcr[j] = np.transpose(n)
             '''
             ycgcr[i,j,0] = np.uint8(n[0,0])
             ycgcr[i,j,1] = np.uint8(n[1,0])
             ycgcr[i,j,2] = np.uint8(n[2,0])

'''             '''
print ycgcr
cv2.waitKey(0)
'''
#混合高斯模型

y,x, channel = ycgcr.shape
like = np.zeros((y,x),np.float64)
M = np.array([[116.6269], [146.2254]])
MM = np.repeat(M,x,1)
'''
print MM
cv2.waitKey(0)
'''
C = np.array([[101.6194, 16.1909],[16.1909, 175.5831]])
C_1 = np.linalg.inv(C)
I = np.identity(x)                 
cgcr = np.zeros((2,x),np.uint8)
for j in range(y):
        cgcr[0,:] = ycgcr[j,:,1]
        cgcr[1,:] = ycgcr[j,:,2]
        like[j] = np.exp(-0.5 * np.sum(np.dot(np.dot(np.transpose(cgcr - MM),C_1),cgcr - MM) * I ,0))     
        
gray = np.zeros((y,x),np.uint8)

#print like
cv2.waitKey(0)

print "min_like",np.amin(like)
cv2.waitKey(0)
#将[0,1]的like转换成[0,255]的gray



print"ok"

if np.amin(like) < 0.1:
    #method 1,单纯用色彩识别有缺陷（以后可以结合基于边缘的图像分割）
    #直接先去噪，再用otsu
    gray = np.uint8(like * 255)
    gray_temp = cv2.medianBlur(gray,5)
    ret_gray,mask = cv2.threshold(gray_temp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print ret_gray
    cv2.waitKey(0)
    cv2.imshow("mask_raw",mask)
    cv2.waitKey(0)



    #修补
    ret,mask = cv2.threshold(mask,0,1,cv2.THRESH_BINARY)

    #闭操作
    kernel = np.ones((25,25),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ret,mask_close = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    cv2.imshow("mask_close",mask_close)
    cv2.waitKey(0)
    '''
    #开操作
    kernel2 = np.ones((25,25),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)
    '''
    ret,mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    cv2.imshow("mask",mask)
    cv2.waitKey(0)





    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2].copy()
    v = cv2.bitwise_and(v,mask)
    hsv[:,:,2] = v.copy()
    output = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
else:
    output = img.copy()
    
cv2.imshow("output",output)
cv2.waitKey(0)


#0-3级静脉分级(粗细,长度,要考虑图像比例!!!!)
gray2 = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
#用gray不用dst，因为高斯滤波延展了腿的边缘
#ret2,dst2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)



#对dst2缩小，去除可能存在背景的边缘部分
ret,dst = cv2.threshold(gray2,0,255,cv2.THRESH_BINARY)


#找出腿，确定比例（滤波所用的核大小）findContours会改变dst2
dst_temp = dst.copy()
contours, hierarchy = \
          cv2.findContours(dst_temp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

i = len(contours)

j = 0
#max_area = -1
#max_num = -1

#[序号][面积]
max_leg = np.array([[-1,-1],[0,0]])

while j < i:
    M = cv2.moments(contours[j])
    area = M['m00']
   
    if(max_leg[1,0] > max_leg[1,1]):
       max_min = max_leg[1,1]
       k = 1
    else:
       max_min = max_leg[1,0]
       k = 0
   
    if(area > max_min):
       max_leg[1,k] = area
       max_leg[0,k] = j
    j += 1
    
print max_leg
cv2.waitKey(0)
#如果面积最大的两块相差小，则认为有2条腿,取重心,opencv函数获得的矩x轴是横轴，y轴是竖轴
x , y = dst.shape
output2 = np.zeros((x,y,3), np.uint8)

if max_leg[1,1] != 0:
    judge = np.float64(max_leg[1,0])/np.float64(max_leg[1,1])

    if( judge > 0.5 and judge < 2):
        leg = 2
        cv2.drawContours(output2, contours, max_leg[0,0], \
                                 (255,255,255), -1)
        #cv2.imshow("output",output)
        #cv2.waitKey(0)
        cv2.drawContours(output2, contours, max_leg[0,1], \
                                 (255,255,255), -1)
    else:
        leg = 1
else:
    leg = 1

if leg == 1:
    if max_leg[1,0] > max_leg[1,1]:
        k = 0
    else:
        k = 1
    cv2.drawContours(output2, contours, max_leg[0,k], \
                                 (255,255,255), -1)

cv2.imshow("output2",output2)
cv2.waitKey(0)

output2 = cv2.cvtColor(output2,cv2.COLOR_BGR2GRAY)


#开操作
kernel2 = np.ones((25,25),np.uint8)
output2 = cv2.morphologyEx(output2, cv2.MORPH_OPEN, kernel2)

ret,output2 = cv2.threshold(output2,0,1,cv2.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
output2 = cv2.erode(output2,kernel,iterations = 1)
ret,output2 = cv2.threshold(output2,0,255,cv2.THRESH_BINARY)

cv2.imshow("out_put2",output2)
cv2.waitKey(0)
  

#print max_num
if max_leg[1,0] >= max_leg[1,1]:
    max_area = max_leg[1,0]
    cnt = contours[max_leg[0,0]]
    plate_lag = max_leg[0,0]
else:
    max_area = max_leg[1,1]
    cnt = contours[max_leg[0,1]]
    plate_lag = max_leg[0,1]
# x,y,w,h = cv2.boundingRect(cnt)


rect = cv2.minAreaRect(cnt)
w,h = rect[1]
angle = rect[2]
print rect
cv2.waitKey(0)
print w,h
cv2.waitKey(0)

Min = min(w,h)#腿的粗细
Max = max(w,h)#腿的长度

print Min
cv2.waitKey(0)

########图片旋转，使腿大致是竖直的#########
'''
rotation_need = 0
if w >= h:
    if angle > - 60 and angle <= 0:
        angle_rotation = 90+angle
        rotation_need = 1

    if angle > 0 and angle < 60:
        angle_rotation = angle - 90
        rotation_need = 1

else:
    if angle <= -30 or angle >= 30:
        angle_rotation = angle
        rotation_need = 1

    
xy_max = max(x,y)
if rotation_need == 1:
    M_rotation = cv2.getRotationMatrix2D((xy_max/2,xy_max/2),angle_rotation,1)
    img=cv2.warpAffine(img,M_rotation,(xy_max,xy_max))
    output2 = cv2.warpAffine(output2,M_rotation,(xy_max,xy_max))

'''

plate = np.zeros((x,y,3),np.uint8)
cv2.drawContours(plate, contours, plate_lag, (255,255,255), -1)
plate = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
cv2.imshow("plate",plate)
cv2.waitKey(0)

xy_have = 0
for j in range(x):
    for i in range(y):
        if plate[j,i] == 255:
            if xy_have == 0:
                x_array = np.array([i],np.int32)
                y_array = np.array([j],np.int32)
                xy_have = 1
            else:
                x_array = np.append(x_array,[i])
                y_array = np.append(y_array,[j])

xy_cov = np.cov(x_array, y_array)
feature_value,feature_vector = np.linalg.eig(xy_cov)

if feature_value[0]>feature_value[1]:
    main_direction = feature_vector[:,0]
else:
    main_direction = feature_vector[:,1]

print "main_direction",main_direction
cv2.waitKey(0)

main_x = main_direction[0]
main_y = main_direction[1]

if main_x < 0:
    main_x = -main_x
    main_y = -main_y
    
if main_y > 2 * main_x\
   or main_y < -2 * main_x:
    rotation_need = 0
else:
    rotation_need = 1


if rotation_need == 1:
    y_axis = np.array([0,1])
    main_direction_len = np.sqrt(np.dot(main_direction,main_direction))
    cos = np.dot(y_axis,main_direction) / main_direction_len
    angel_now = np.arccos(cos)
    degree = 180.0 / np.pi * angel_now
    print "degree",degree
    if main_direction[0] * main_direction[1] >= 0:
        order_rotation = -1
    else:
        order_rotation = 1
    if degree < 90:
        angle_rotation =  degree
    else:
        angle_rotation = 180 - degree

    angle_rotation = angle_rotation * order_rotation
    print "angle_rotation",angle_rotation
    xy_max = max(x,y)

    M_rotation = cv2.getRotationMatrix2D((xy_max/2,xy_max/2),angle_rotation,1)
    img=cv2.warpAffine(img,M_rotation,(xy_max,xy_max))
    output2 = cv2.warpAffine(output2,M_rotation,(xy_max,xy_max))

#####################################
mask_output2 = output2.copy()
ret_temp,output2_temp  = cv2.threshold(output2,0,1,cv2.THRESH_BINARY)



    
    
    
gray3 =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray3 =  cv2.bitwise_and(gray3,mask_output2)
cv2.imshow("gray3",gray3)
cv2.waitKey(0)

#腿部亮度归一化
def normalization(pic,mask):
    cv2.normalize(pic,pic,255,0,cv2.NORM_MINMAX,-1,mask)
    return pic
'''
gray3 = normalization(gray3,mask_output2)
cv2.imshow("gray3_normalization",gray3)
cv2.waitKey(0)
'''
#将图片切割,计算均值方差

'''
means = np.zeros((piece1,piece2),np.float64)
standards = np.zeros((piece1,piece2),np.float64)
contrasts = np.zeros((piece1,piece2,4),np.float64)#分别是0,45,90,135
ASMs = np.zeros((piece1,piece2),np.float64)#只取水平方向

C = np.zeros((piece1,piece2),np.float64)#相对对比度

S = np.zeros((piece1,piece2),np.float64)#相对标准差

M = np.zeros((piece1,piece2),np.float64)#相对均值
'''
#shape得到的是（y,x）
y,x = gray3.shape
print x,y
cv2.waitKey(0)
'''
stepx = np.uint32(Min / 10)
stepy = np.uint32(Min / 10)
'''


value = max_area / Max / 6

stepx = np.uint32(value)
stepy = np.uint32(value)
print "value",value
piece1 = y / stepy
piece2 = x / stepx

distance =  stepx / 25 + 1
print stepx,stepy
print piece1,piece2
print "distance",distance
cv2.waitKey(0)
'''
#算均值标准差
def cal(m):
    result = np.zeros((2),np.float64)
    result[0] = np.mean(m)
    result[1] = np.std(m)
    return result
'''

#判断方格是否在腿内
def contain(j,i):
    matrix =  output2_temp[(j) * stepy : (j + 1) * stepy,(i) * stepx : (i + 1) * stepx ]
    result = np.sum(matrix)
    #print result
    
    #cv2.waitKey(0)
    if result == stepy * stepx:
        return 1
    else:
        return 0
#算灰度共生矩阵

row = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
r = np.repeat(row,8,1)
        

column = np.array([[1,2,3,4,5,6,7,8]])
c = np.repeat(column,8,0) 

r_c_square= (r - c)**2
    
dst = gray3  
lag = 0    
for j in range(piece1):
    for i in range(piece2):
      if contain(j,i) == 1:
        m = dst[(j) * stepy : (j + 1) * stepy,(i) * stepx : (i + 1) * stepx ]
        if lag == 0:
            pieces = np.array([[j,i]], np.uint8)
            means = np.array([np.mean(m)])
            standards = np.array([np.std(m)])
        else:
            pieces = np.append(pieces, [[j,i]], axis = 0)
            means = np.append(means,[np.mean(m)])
            standards = np.append(standards, [np.std(m)])
            #means[j,i],standards[j,i] = cal(m)

        #recoding
        #reduce graylevels to 1-8
        n = np.zeros((stepy + 2 * distance,stepx + 1 * distance),np.uint8)
        '''
        for a in range(stepy):
            for b in range(stepx):
                n[a + 1,b] = np.uint8( m[a,b] / 32 + 1  )
        '''

        n[1*distance:stepy+1*distance,0:stepx] = np.uint8( m / 32 + 1  )
        #horizontal right_diagonal vertical left_diagonal
        co_horizontal = np.zeros((8,8),np.int32)
        co_right_diagonal = np.zeros((8,8),np.int32)
        co_vertical = np.zeros((8,8),np.int32)
        co_left_diagonal = np.zeros((8,8),np.int32)

        for a in range(stepy):
            for b in range(stepx):
                if n[a+1*distance,b] * n[a,b+1*distance] != 0:
                    co_right_diagonal[n[a+1*distance,b] - 1 ,n[a,b+1*distance] - 1] += 1
                    co_right_diagonal[n[a,b+1*distance] - 1 ,n[a+1*distance,b] - 1] += 1
                if n[a+1*distance,b] * n[a+1*distance,b+1*distance] !=0:
                    co_horizontal[n[a+1*distance,b] - 1 ,n[a+1*distance,b+1*distance] - 1] += 1
                    co_horizontal[n[a+1*distance,b+1*distance] - 1 ,n[a+1*distance,b] - 1] += 1
                if n[a+1*distance,b] * n[a+2*distance,b+1*distance] !=0:
                    co_left_diagonal[n[a+1*distance,b] - 1 ,n[a+2*distance,b+1*distance] - 1] += 1
                    co_left_diagonal[n[a+2*distance,b+1*distance] - 1 ,n[a+1*distance,b] - 1] += 1
                if n[a+1*distance,b] * n[a+2*distance,b] !=0:
                    co_vertical[n[a+1*distance,b] - 1 ,n[a+2*distance,b] - 1] += 1
                    co_vertical[n[a+2*distance,b] - 1 ,n[a+1*distance,b] - 1] += 1
        #print j,i            
        #print co_horizontal
        '''
        print co_right_diagonal
        print co_vertical
        print co_left_diagonal
        '''
        #归一化
        p_h = co_horizontal / np.float64(np.sum(co_horizontal))
        p_r = co_right_diagonal / np.float64(np.sum(co_right_diagonal))
        p_l = co_left_diagonal / np.float64(np.sum(co_left_diagonal))
        p_v = co_vertical / np.float64(np.sum(co_vertical))
        '''
        #均值
        gi = np.arange(1,9)
        gj = np.arange(1,9)
        pi = np.sum(p,axis = 1)
        pj = np.sum(p,axis = 0)
        mr = np.dot(gi,pi)
        mc = np.dot(gj,pj)

        temp_r = (gi - mr)**2
        va_r = np.dot(temp_r,pi)
        temp_c = (gj - mc)**2
        va_c = np.dot(temp_c,pj)
        '''
        
        '''
        #相关度
        stand = np.sqrt(va_r * va_c)
        if stand != 0:
               relative = np.dot(gi - mr, np.dot(p, gj - mc)) / np.sqrt(va_r * va_c)
               print "relative =" ,relative
        else:
               print "均匀"
        '''
        #对比度,溃烂区域的对比度很大（对比度大是识别溃烂的必要不充分条件）
        contrast_h = 0
        contrast_r = 0
        contrast_v = 0
        contrast_l = 0
       
        '''    
        for a in range(8):
               for b in range(8):
                      contrast_h += (a - b)**2 * p_h[a,b]
                      contrast_r += (a - b)**2 * p_r[a,b]
                      contrast_v += (a - b)**2 * p_v[a,b]
                      contrast_l += (a - b)**2 * p_l[a,b]
        '''
        contrast_h = np.sum(r_c_square * p_h)
        contrast_r = np.sum(r_c_square * p_r)
        contrast_v = np.sum(r_c_square * p_v)
        contrast_l = np.sum(r_c_square * p_l)
    
        if lag == 0:
            #contrasts_v = np.array([contrast_v])
            contrasts = np.array([np.average([contrast_h, contrast_r, contrast_v, contrast_l])])
            #lag = 1
        else:
            #contrasts_v = np.append(contrasts_v,[contrast_v])
            contrasts = np.append(contrasts,[np.average([contrast_h, contrast_r, contrast_v, contrast_l])])
        #print "contrast =", contrast
        
        #一致性
        #ASMs[j,i] = np.sum(p_h**2)
        #print "ASM =",ASM
        '''    
        #一致性（能量）
        ASM_h = 0
        ASM_r = 0
        ASM_v = 0
        ASM_l = 0

        for a in range(8):
               for b in range(8):
                      ASM_h += p_h[a,b]**2
                      ASM_r += p_r[a,b]**2
                      ASM_v += p_v[a,b]**2
                      ASM_l += p_l[a,b]**2
        if lag == 0:
            ASMs = np.array([np.average([ASM_h, ASM_r, ASM_v, ASM_l])])
            lag = 1
        else:
            ASMs = np.append(ASMs,[np.average([ASM_h, ASM_r, ASM_v, ASM_l])] )


        
ASMs_avg = np.average(ASMs)
ASMs_std = np.std(ASMs)
        '''
        p_h += 0.000000001
        p_r += 0.00000001
        p_v += 0.00000001
        p_l += 0.00000001

        #熵
        entropy_h = - np.sum(p_h * np.log2(p_h))
        entropy_r = - np.sum(p_r * np.log2(p_r))
        entropy_v = - np.sum(p_v * np.log2(p_v))
        entropy_l = - np.sum(p_l * np.log2(p_l))

        if lag == 0:
            entropys = np.array([np.average([entropy_h, entropy_r, entropy_v, entropy_l])])
            lag = 1
        else:
            entropys = np.append(entropys,[np.average([entropy_h, entropy_r, entropy_v, entropy_l])] )

entropy_avg = np.average(entropys)
entropy_std = np.std(entropys)
'''
contrast_v_avg  =  np.average(contrasts_v)
contrast_v_std = np.std(contrasts_v)
'''       
contrast_avg  =  np.average(contrasts)
contrast_std = np.std(contrasts)

standards_avg = np.average(standards)
standards_std = np.std(standards)

means_avg = np.average(means)
means_std = np.std(means)
'''
for a in range(piece1):
       for b in range(piece2):
              C[a,b] = (np.average(contrasts[a,b]) - contrast_avg) / contrast_std
              S[a,b] = (standards[a,b] - standards_avg) / standards_std
              M[a,b] = (means[a,b] - means_avg) / means_std
'''
C = (contrasts - contrast_avg) / contrast_std
#C_v = (contrasts_v - contrast_v_avg) / contrast_v_std
S = (standards - standards_avg) / standards_std
M = (means - means_avg) / means_std
#A = (ASMs - ASMs_avg) / ASMs_std
E = (entropys - entropy_avg) / entropy_std
print pieces
print means
print standards
print contrasts

#print ASMs
print C
print S
print M
#print A
cv2.waitKey(0)

img2 = img.copy()
img3 = img.copy()
img4 = img.copy()
number = np.alen(M)
    
for i in range(piece2):
       cv2.line(img2,(i * stepx,0),(i * stepx,y),(0,255,0),1)

for j in range(piece1):
        cv2.line(img2,(0,j * stepy),(x,j * stepy),(0,255,0),1)
        
for i in range(piece2):
       cv2.line(img3,(i * stepx,0),(i * stepx,y),(0,255,0),1)

for j in range(piece1):
        cv2.line(img3,(0,j * stepy),(x,j * stepy),(0,255,0),1)

for i in range(piece2):
       cv2.line(img4,(i * stepx,0),(i * stepx,y),(0,255,0),1)

for j in range(piece1):
        cv2.line(img4,(0,j * stepy),(x,j * stepy),(0,255,0),1)

      


hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

print number
cv2.waitKey(0)

h_avg = np.zeros((number),np.float64)
s_avg = np.zeros((number),np.float64)
for a in range(number):
              j,i = pieces[a]
              #对比度大&&标准差大
              h = hsv[stepy * j : stepy * (j + 1), stepx * i : stepx * (i + 1), 0]
              s = hsv[stepy * j : stepy * (j + 1), stepx * i : stepx * (i + 1), 1]
              h_avg[a] = np.average(h)
              s_avg[a] = np.average(s)

for a in range(number):
    j,i = pieces[a]
    if h_avg[a] < 8.5:
        cv2.rectangle(img3,((i) * stepx,(j ) * stepy),\
                            ((i + 1) * stepx - 1,(j + 1) * stepy - 1),(0,0,255),1)
        
cv2.imshow("img3",img3)
cv2.waitKey(0)

kernel = np.ones((5,5),np.uint8)
mask_temp = cv2.erode(mask_output2,kernel,iterations = 1)
gray4 =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray4 =  cv2.bitwise_and(gray4,mask_temp)
cv2.imshow("gray4",gray4)
cv2.waitKey(0)
'''
#腿部亮度归一化到50-170
cv2.normalize(gray4,gray4,210,20,cv2.NORM_MINMAX,-1,mask_temp)
cv2.imshow("gray4_normalization",gray4)
cv2.waitKey(0)
'''
'''
gray4 = normalization(gray4,mask_temp)
cv2.imshow("gray4_normalization",gray4)
cv2.waitKey(0)
'''
gray3_2 = cv2.medianBlur(gray3,3)
gray4_2 = cv2.medianBlur(gray4,3)

edges1 = cv2.Canny(gray3_2,150,200)
edges2 = cv2.Canny(gray4_2,150,200)



edges_temp = cv2.bitwise_and(edges1,edges2)
cv2.imshow("edges",edges_temp)

ret,edges = cv2.threshold(edges_temp,0,1,cv2.THRESH_BINARY)
y,x = edges.shape
x_sum = 0
y_sum = 0
for j in range(y):
    for i in range(x):
        x_sum += i * edges[j,i]
        y_sum += j * edges[j,i]
flag = np.sum(edges)
if flag < 30:
    flag = 0
if flag != 0:
    
    x_centroid = x_sum / np.sum(edges)
    y_centroid = y_sum / np.sum(edges)
    print "x_centroid,y_centroid",x_centroid,y_centroid
print x_sum,y_sum
print np.sum(edges)
#
cv2.waitKey(0)
cv2.waitKey(0)
cv2.waitKey(0)

'''
while edges[y_centroid, x_centroid] == 1:
    x_centroid += 1

def inner(temp,e,x,y,w):
    temp2 = temp.copy()
    cv2.rectangle(temp2,(x-w,y-w),(2*w,2*w),(255,255,255),-1)
    temp3 = cv2.cvtColor(temp2,cv2.COLOR_BGR2GRAY)
    
    interect = cv2.bitwise_and(temp3, e)
    result = np.sum(interect)
    if result == 0:
        return 0
    else:
        return 1
'''
width = flag / 8
'''
temp = np.zeros((y,x,3),np.uint8)
while inner(temp,edges_temp,x_centroid,y_centroid,width) == 0:
    width += 1
'''
#width -= 1
h = hsv[:,:,0]
'''
h_sum = 0
h_average = 100
if flag != 0:
    for j in range(y_centroid - width, y_centroid + width + 1):
        for i in range(x_centroid - width, x_centroid + width + 1):
            h_sum += h[j,i]

    h_average = h_sum / (2 * width)**2
    print "h_average" , h_average
print "width", width

cv2.waitKey(0)
cv2.waitKey(0)


cv2.waitKey(0)
rank5 = 0
'''
    #cv2.rectangle(img2,(x-width,y-width),(2*width,2*width),(0,0,255),1)
    #cv2.rectangle(img2,(x_centroid-width,y_centroid-width),(x_centroid+width,y_centroid+width),(0,0,255),1)
    # print "6级"
    
s_average = np.average(s_avg)
s_standard = np.std(s_avg)
s_relative = (s_avg - s_average)/s_standard
'''
h_min = 30

h_max = 30
'''
edge_flag = np.zeros((number),np.uint8)
edges_area = np.zeros((number),np.int32)
edges_inner = np.zeros((number),np.int32)
h = hsv[:,:,0]
s = hsv[:,:,1]
I = hsv[:,:,2]
points = 0
'''
y
x
color: white 0 gray 1 black 2

start_time
end_time

'''
court = np.zeros((number,stepy,stepx,5),np.int32)

for a in range(number):
        j,i = pieces[a]
        for m in range(0,stepy):
          for n in range(0,stepx):
            court[a,m,n,0] = m
            court[a,m,n,1] = n
            court[a,m,n,2] = 0
           
            
            
points = 0

###DFS-VISIT
def DFS_VISIT_2(a,m,n,kind):
  global points
  j,i = pieces[a]
  u = j*stepy+m
  v = i*stepx+n
  if kind == 1 and (h[u,v] <= 3 or h[u,v] >= 170) and (s[u,v] >= 150 and I[u,v] >= 170 or s[u,v] >= 200 and I[u,v] >= 130)\
     or kind == 2 and  h[u,v] >= 4  and  h[u,v] <=8 and  s[u,v] >= 75 and s[u,v] <= 170 and I[u,v] >= 90\
     or kind == 3 and ((h[u,v] >= 12 and h[u,v]<= 28 or h[u,v] >= 130 and h[u,v] <= 150) and I[u,v] >= 60 or (h[u,v] <= 3 or h[u,v] >= 170) and s[u,v] <= 100 and I[u,v] >= 120)\
     or kind == 4 and ((h[u,v] >= 5 and h[u,v]<= 28 or h[u,v] >= 130 and h[u,v] <= 150) and s[u,v] <= 170 and s[u,v] >= 75 and I[u,v] >= 60 or (h[u,v] <= 3 or h[u,v] >= 170) and s[u,v] <= 110 and I[u,v] >= 120):
    points += 1
    court[a,m,n,3] = points
    court[a,m,n,2] = 1
    if n > 0:
        
            if court[a,m,n-1,2] == 0:
               
              
               DFS_VISIT_2(a,m,n-1,kind)
                
    if n < stepx - 1:
        
            if court[a,m,n+1,2] == 0:
              
                DFS_VISIT_2(a,m,n+1,kind)
    if m > 0:
        
            if court[a,m-1,n,2] == 0:
                
                DFS_VISIT_2(a,m-1,n,kind)
                
    if m < stepy - 1:
        
            if court[a,m+1,n,2] == 0:
                
                DFS_VISIT_2(a,m+1,n,kind)
    
    court[a,m,n,2] = 2
    points += 1
    court[a,m,n,4] = points


for a in range(number):

    j,i = pieces[a]
    edge_length = np.sum(edges[stepy * j : stepy * (j + 1), stepx * i : stepx * (i + 1)] )
    if edge_length > stepx / 2.5 :
        edge_flag[a] = 1
        #edge_point = 0
        inner = 0
        
        for m in range(0,stepy):
            for n in range(0,stepx):
                if court[a,m,n,2] == 0:
                    DFS_VISIT_2(a,m,n,1)
                    if court[a,m,n,4] / 2 >= stepx * stepy / 50:
                        edges_inner[a] += court[a,m,n,4] / 2
                    points = 0
        print "j,i",j,i
        print"edges_inner",edges_inner[a]
        print"stepx * stepy / 10",stepx * stepy / 10
        cv2.waitKey(0)
        if edges_inner[a]>= stepx * stepy / 20:
            for m in range(0,stepy):
                for n in range(0,stepx):
                    if court[a,m,n,2] == 0:
                        DFS_VISIT_2(a,m,n,4)
                        if court[a,m,n,4] / 2 >= stepx * stepy / 50:
                            edges_area[a] += court[a,m,n,4] / 2
                        points = 0
            print"edges_area",edges_area[a]
            cv2.waitKey(0)
        else:
            for m in range(0,stepy):
                for n in range(0,stepx):
                    if court[a,m,n,2] == 0:
                        DFS_VISIT_2(a,m,n,2)
                        if court[a,m,n,4] / 2 >= stepx * stepy / 50:
                            edges_inner[a] += court[a,m,n,4] / 2
                        points = 0
            print"edges_inner_2",edges_inner[a]
            cv2.waitKey(0)
            for m in range(0,stepy):
                for n in range(0,stepx):
                    if court[a,m,n,2] == 0:
                        DFS_VISIT_2(a,m,n,3)
                        if court[a,m,n,4] / 2 >= stepx * stepy / 50:
                            edges_area[a] += court[a,m,n,4] / 2
                        points = 0
            

                        
                    
                    



        """
        for m in range(stepy * j , stepy * (j + 1)):
            for n in range(stepx * i , stepx * (i + 1)):
                if h[m,n] >= 12 and h[m,n] <= 28 and I[m,n] >= 80:
                    edges_area[a] += 1
                if (h[m,n] <= 8 or h[m,n] >=170)and s[m,n]>75 and I[m,n] >= 80:
                    edges_inner[a] += 1
        
        """
        '''
        if h_avg[a] < h_min:
            h_min = h_avg[a]
        if h_avg[a] < h_min:
            h_min = h_avg[a]
        '''
'''
print "h_min",h_min
print "h_max",h_max
'''
mean_wrong = np.array([[ 117.4051625],[ 143.4328375]])
mean_right = np.array([[ 117.8874    ],[ 145.67651429]])
cov_wrong = np.array([[ 19.09419453 ,-25.93495621],[-25.93495621 , 40.36949382]])
cov_right = np.array([[ 16.42427016, -22.00842461],[-22.00842461 , 36.7027956 ]])

if rotation_need == 1:
    #bgr转ycgcr
    b = img[:,:,0].copy()
    g = img[:,:,1].copy()
    r = img[:,:,2].copy()
    AA = np.array([[65.481, 128.553, 24.966],[-81.085, 112, -30.915],[112, -93.768, -18.214]])
    BB = np.array([[16],[128],[128]])
    CC = np.repeat(BB,x,1)
             
    ycgcr = np.zeros(img.shape,np.uint8)

    for j in range(y):
             
             n = CC + 1.0/256 * np.dot(AA,[r[j],g[j],b[j]])
             ycgcr[j] = np.transpose(n)
           
#混合高斯模型


#wrong
y,x, channel = ycgcr.shape
like_wrong = np.zeros((y,x),np.float64)
MM = mean_wrong
MM = np.repeat(MM,x,1)
   
CC = cov_wrong
 
C_1 = np.linalg.inv(CC)
I = np.identity(x)                 
cgcr = np.zeros((2,x),np.uint8)
for j in range(y):
        cgcr[0,:] = ycgcr[j,:,1]
        cgcr[1,:] = ycgcr[j,:,2]
        like_wrong[j] = np.exp(-0.5 * np.sum(np.dot(np.dot(np.transpose(cgcr - MM),C_1),cgcr - MM) * I ,0))  


#right
y,x, channel = ycgcr.shape
like_right = np.zeros((y,x),np.float64)
MM = mean_right
MM = np.repeat(MM,x,1)
    

CC = cov_right
C_1 = np.linalg.inv(CC)
I = np.identity(x)                 
cgcr = np.zeros((2,x),np.uint8)
for j in range(y):
        cgcr[0,:] = ycgcr[j,:,1]
        cgcr[1,:] = ycgcr[j,:,2]
        like_right[j] = np.exp(-0.5 * np.sum(np.dot(np.dot(np.transpose(cgcr - MM),C_1),cgcr - MM) * I ,0))


like_temp = like_wrong - like_right



def judge_rank_6(p):
    m,n = pieces[p]
    spots_6 = 0
    for u in range(stepy * m , stepy * (m + 1)):
        for v in range(stepx * n , stepx * (n + 1)):
            if hsv[u,v,1] > 80 and (hsv[u,v,0] <= 8 or hsv[u,v,0]>= 170) and hsv[u,v,2]>85:
                spots_6 += 1
    print "stepy * stepx * 2 / 3",stepy * stepx * 2 / 3
    print "spots_6",spots_6
    if spots_6 > stepy * stepx * 2 / 3:
        return 1
    else:
        return 0

def black_rank_5_edge(p):
    m,n = pieces[p]
    black_edge = 0
    if edge_flag[p] == 1:
     for u in range(stepy * m , stepy * (m + 1)):
        for v in range(stepx * n , stepx * (n + 1)):
            if hsv[u,v,1] < 60 and hsv[u,v,0]>= 110 and hsv[u,v,2]< 75:
                black_edge += 1
     if black_edge > stepy * stepx  / 20:
        return 1
     else:
        return 0
    else:
        return 0
print "value",value                
            
rank = np.zeros((number),np.uint8)

rank_6 = 0

area_judge = stepx * stepy / 15
inner_judge = stepx * stepy / 20

if flag != 0 : 
  for a in range(number):
    #print  "C,S",C[a],S[a]
    j,i = pieces[a]
    print "j,i",j,i
    print  "C,S,E",C[a],S[a],E[a]
    #if np.sum(edges[stepy * j : stepy * (j + 1), stepx * i : stepx * (i + 1)] ) > 0:
    #if flag != 0:       
    print h_avg[a],s_avg[a]
    print "stepx * stepy / 15",stepx * stepy / 15
    print "edges_area[a]",edges_area[a]
    print "edges_inner[a]",edges_inner[a]
    #if s_avg[a] > 80 and (h_avg[a] >= 80 or h_avg[a] <= 8.5)  :
    
    if judge_rank_6(a) == 1:
        
        cv2.waitKey(0)
        '''
        if edges_area[a] >= stepx * stepy / 15:
            print j,i
            print "6级"

            rank[6] = 6
        else:
        '''
        edges_6 = 0
        for b in range(number):
                    m, n = pieces[b]
                    #print "edges_area[b]",edges_area[b]
                    #print "stepx * stepy / 30",stepx * stepy / 30
                    #cv2.waitKey(0)
                    if m  == j - 1  and n == i and   (edges_area[b] >= area_judge and edges_inner[b] >= inner_judge and S[b] > 0.5 ):
                        edges_6 += 1
                    if m  == j - 1  and n == i - 1  and   (edges_area[b] >= area_judge and edges_inner[b] >= inner_judge and S[b] > 0.5 ):
                        edges_6 += 1
                    if m  == j   and n == i - 1 and   (edges_area[b] >= area_judge and edges_inner[b] >= inner_judge and S[b] > 0.5 ):
                        edges_6 += 1
                    if m  == j + 1   and n == i - 1 and   (edges_area[b] >= area_judge and edges_inner[b] >= inner_judge and S[b] > 0.5 ):
                        edges_6 += 1
                    if m  == j   and n == i + 1 and   (edges_area[b] >= area_judge and edges_inner[b] >= inner_judge and S[b] > 0.5 ):
                        edges_6 += 1
                    if m  == j + 1   and n == i + 1 and   (edges_area[b] >= area_judge and edges_inner[b] >= inner_judge and S[b] > 0.5 ):
                        edges_6 += 1
                    if m  == j - 1   and n == i + 1 and   (edges_area[b] >= area_judge and edges_inner[b] >= inner_judge and S[b] > 0.5 ):
                        edges_6 += 1
                    if m  == j + 1  and n == i  and   (edges_area[b] >= area_judge and edges_inner[b] >= inner_judge and S[b] > 0.5 ):
                        edges_6 += 1

 

        if edges_6 >= 2:    
                             print j,i
                             print "6级"
                             rank[a] = 6
                             
                             
                            
                        
        cv2.waitKey(0)
             
       
    
    else :
        
            if C[a] > 3 and S[a] > 1.5  and E[a] > 2 and h_avg[a] > 11 \
               and edges_inner[a]<inner_judge and (edges_area[a] >= area_judge or black_rank_5_edge(a) == 1):#and h_avg[a] > 10:

               
                    rank[a] = 5
                    print "maybe 5级1"
                    cv2.waitKey(0)
                    
            else:
                if C[a] > 2 and S[a] > 1.5   and E[a] > 1.5 and h_avg[a] > 11 \
                 and edges_inner[a]<inner_judge :#and h_avg[a] > 10:
                  for b in range(number):
                    m, n = pieces[b]
                    #print "edges_area[b]",edges_area[b]
                    #print "stepx * stepy / 30",stepx * stepy / 30
                    #cv2.waitKey(0)
                    print "m,n",m,n
                    print "black_rank_5_edge ",black_rank_5_edge(b)
                    if(m  == j - 1  and n == i\
                       or m  == j - 1  and n == i - 1  \
                       or m  == j   and n == i - 1 \
                       or m  == j + 1   and n == i - 1 \
                       or m  == j   and n == i + 1 \
                       or m  == j + 1   and n == i + 1 \
                       or m  == j - 1   and n == i + 1 \
                       or m  == j + 1  and n == i ) \
                       and (edges_area[b] >= area_judge  or black_rank_5_edge(b) == 1):
                        rank[a] = 5

                        
                        print C[a],S[a],
                        
                        print "maybe 5级2"
                        
                        cv2.waitKey(0)

                        break
                    
  for a in range(number):
   
                    j,i = pieces[a]

        
                    print j,i
                    print "stepx * stepy / 15",stepx * stepy / 15
                    print "edges_area[a]",edges_area[a]
                    print "edges_inner[a]",edges_inner[a]
                    if rank[a] != 6 and rank[a]!= 5 and edges_area[a] >= area_judge and  edges_inner[a] >= inner_judge and S[a]>0.5 :
           
                          
                        print j,i
                        print s_relative[a]          
                        print "maybe  6级 "
                        print "edges_area[a]",edges_area[a]
                        print "edges_inner[a]",edges_inner[a]
                        
                        rank[a] = 7
                          
                        cv2.waitKey(0)
                        

binc = np.bincount(rank)
print "binc",binc
if np.alen(binc) == 8:
    if binc[6] == 0 and binc[5] > 0:
        for a in range(number):
            if rank[a] == 7:
                rank[a] = 5
    else :
        for a in range(number):
            if  rank[a] == 7:
                rank[a] = 6


            
for a in range(number):
        j,i = pieces[a]
    
        if rank[a] == 0:
                      if M[a] < -1 :#灰度值很低，但是也需要颜色位置信息辅助，否则阴影和黑斑分不开
                        '''
                       black = 0
                       for m in range(j*stepy, (j+1)*stepy):
                        for n in range(i*stepx, (i+1)*stepx):
                         if like_temp[m,n] > 0:
                                   black += 1
                                   
                
                       if black > stepy * stepx /5:
                        '''
                        print j,i
                        print a
                        print "M",M[a]
                        print "S",S[a]
                        print "maybe 4级"
                        rank[a] = 4
                        
                        cv2.rectangle(img4,((i) * stepx,(j ) * stepy),\
                                       ((i + 1) * stepx - 1,(j + 1) * stepy - 1),(255,0,0),1)
                        
                        cv2.waitKey(0)
                        
                        
                      
                      if S[a] > 1.2:#灰度值很低，但是也需要颜色位置信息辅助，否则阴影和黑斑分不开
                            print j,i
                            print a
                            print S[a]
                            print "4级edge"
                            rank[a] = 8
                            cv2.rectangle(img4,((i) * stepx,(j ) * stepy),\
                                       ((i + 1) * stepx - 1,(j + 1) * stepy - 1),(0,255,255),1)
                            

cv2.imshow("img4",img4)
cv2.waitKey(0)

'''
cv2.imshow("img2",img2)
cv2.waitKey(0)
print "rank_6",rank_6
print "rank_5",rank_5
'''
'''
if rank_6 == 0 and rank_5 == 1:
    for a in range(number):
        j,i = pieces[a]
        if rank[a] == 7:
            rank[a] = 5
            print j,i
            print "5级"
            cv2.rectangle(img2,((i) * stepx,(j ) * stepy),\
                                       ((i + 1) * stepx - 1,(j + 1) * stepy - 1),(255,0,255),1)
else:
    for a in range(number):
        j,i = pieces[a]
        if rank[a] == 7:
            rank[a] = 6
            print j,i
            print "6级"
            cv2.rectangle(img2,((i) * stepx,(j ) * stepy),\
                                       ((i + 1) * stepx - 1,(j + 1) * stepy - 1),(0,0,255),1)
'''

'''         
#去除灰度值低但是很细的地方，因为黑斑是一大块的，边缘阴影很窄，需要考虑腿的粗细
width_thred = np.int32( Min / 100 )         
print " 最窄黑斑限制",width_thred


checked = np.zeros((number),np.uint8)

for a in range(number):
   j,i = pieces[a] 
   if rank[a] == 4 and checked[a] == 0:
       team = np.array([a])
       checked[a] = 1
       count = 1
       b = a + 1
       if b < number:
           m,n = pieces[b]
           while m == j  and n == i + 1 and rank[b] == 4:
               checked[b] = 1
               count += 1
               team = np.append(team,[b])
               j = m
               i = n
               b += 1
               if b>= number:
                   
                   break;
               else:
                   m,n = pieces[b]
       print count
       cv2.waitKey(0)
       if count < width_thred:
           team_length = np.alen(team)
           for i in range (team_length):
               rank[team[i]] = 0
'''
'''
maybe_black_edge = 0
real_black_edge = 0


for a in range(number):
    if rank[a] == 8:
        maybe_black_edge += 1
        j,i = pieces[a]
        for b in range(number):
            m,n = pieces[b]
            if ( m == j - 1 and n == i\
                   or m == j  and n == i - 1\
                   or m == j + 1 and n == i\
                   or m == j  and n == i + 1) and rank[b] == 4:
                real_black_edge += 1

if maybe_black_edge != 0:

    judge = np.float(real_black_edge) / maybe_black_edge
    print "judge",judge
    cv2.waitKey(0)
'''

#去除灰度值低但是很细的地方，因为黑斑是一大块的，边缘阴影很窄，需要考虑腿的粗细



width_thred = 2
print "max_area / Max",max_area / Max
print "Min",Min
print " 最窄黑斑限制",width_thred


checked = np.zeros((number),np.uint8)

for a in range(number):
       j,i = pieces[a] 
       if rank[a] == 4  and checked[a] == 0:
           team = np.array([a])
           checked[a] = 1
           count = 1
           b = a + 1
           if b < number:
               m,n = pieces[b]
               while m == j  and n == i + 1 and rank[b] == 4:
                   checked[b] = 1
                   count += 1
                   team = np.append(team,[b])
                   j = m
                   i = n
                   b += 1
                   if b>= number:
                   
                       break;
                   else:
                       m,n = pieces[b]
           print count
           cv2.waitKey(0)
           if count < width_thred:
               team_length = np.alen(team)
               for i in range (team_length):
                  
                       rank[team[i]] = 0

#######成块的黑斑最少4个#############

'''
y
x
color: white 0 gray 1 black 2
father_y
father_x
start_time
end_time
number
'''
valid = np.zeros((piece1,piece2,8),np.int32)
temp = np.zeros((piece1,piece2),np.uint8)
valid -= 1

for a in range(number):
        j,i = pieces[a]
        if rank[a] == 4:
            valid[j,i,0] = j
            valid[j,i,1] = i
            valid[j,i,2] = 0
            valid[j,i,3] = -1
            valid[j,i,4] = -1
            valid[j,i,7] = a
            temp[j,i] = 1 
time = 0

###DFS-VISIT
def DFS_VISIT(m,n):
    global time
    time += 1
    valid[m,n,5] = time
    valid[m,n,2] = 1
    if valid[m,n,1] > 0:
        
            if valid[m,n-1,2] == 0:
                valid[m,n-1,3] = m
                valid[m,n-1,4] = n
                DFS_VISIT(m,n-1)
                
    if valid[m,n,1] < piece2 - 1:
        
            if valid[m,n+1,2] == 0:
                valid[m,n+1,3] = m
                valid[m,n+1,4] = n
                DFS_VISIT(m,n+1)
    if valid[m,n,0] > 0:
        
            if valid[m-1,n,2] == 0:
                valid[m-1,n,3] = m
                valid[m-1,n,4] = n
                DFS_VISIT(m-1,n)
                
    if valid[m,n,0] < piece1 - 1:
        
            if valid[m+1,n,2] == 0:
                valid[m+1,n,3] = m
                valid[m+1,n,4] = n
                DFS_VISIT(m+1,n)
    
    valid[m,n,2] = 2
    time += 1
    valid[m,n,6] = time


    
###DFS
for J in range(piece1):
    for I in range(piece2):
        if valid[J,I,2] == 0:
            print "J,I",J,I
            
            DFS_VISIT(J,I)
            print "valid[J,I,6]",valid[J,I,6]
            cv2.waitKey(0)
            if valid[J,I,6] / 2  >= 4:
                for u in range(piece1):
                  for v in range(piece2):
                    if temp[u,v] == 1 and valid[u,v,3] != -1:
                        temp[u,v] = 2
                temp[J,I] = 2
                
                maybe_edge_black_up = 0
                real_edge_black_up = 0
                judge_up = 0
                '''
                for a in range(number):
                    j,i = pieces[a]
                    if rank[a] == 4:
                        for b in range(number):
                            m,n = pieces[b]
                
                            if ( m == j - 1 and n == i\
                               #or m == j  and n == i - 1\
                               #or m == j  and n == i + 1\
                               #or m == j + 1 and n == i
                               ) and rank[b] != 4 and rank[b] != 5 and rank[b]!=6:
                        
                                maybe_edge_black_up += 1
                           
                                if rank[b] == 8 or M[b]> - 0.3:
                                    real_edge_black_up += 1
                            
                                break
                '''
                break_point = 0
                for j in range(piece1):
                    for i in range(piece2):
                        if temp[j,i] == 2:
                          if j > 1:
                            for b in range(number):
                                m,n = pieces[b]
                
                                if ( m == j - 1 and n == i \
                               ) and rank[b] != 4 and rank[b] != 5 and rank[b]!=6:
                        
                                    maybe_edge_black_up += 1
                           
                                    if rank[b] == 8 or M[b]> -0.3:
                                        real_edge_black_up += 1
                            
                                    break
                            
                          else:
                               maybe_edge_black_up = 0
                               break_point = 1
                               break
                    if break_point == 1:
                        break
                            
                if maybe_edge_black_up > 1:                           
                    judge_up = np.float(real_edge_black_up) / maybe_edge_black_up
                    print "maybe_edge_black_up",maybe_edge_black_up
                    print "real_edge_black_up",real_edge_black_up
                    print "judge_up",judge_up
                    cv2.waitKey(0)

                maybe_edge_black_down = 0
                real_edge_black_down = 0
                judge_down = 0
                '''
                for a in range(number):
                    j,i = pieces[a]
                    if rank[a] == 4:
            
                        for b in range(number):
                            m,n = pieces[b]
                
                            if ( m == j + 1 and n == i\
                               #or m == j  and n == i - 1\
                               #or m == j  and n == i + 1\
                               #or m == j + 1 and n == i
                               ) and rank[b] != 4 and rank[b] != 5 and rank[b]!=6:
                        
                                maybe_edge_black_down += 1
                            
                                if rank[b] == 8  or M[b]> - 0.3:
                                    real_edge_black_down += 1
                                break
                '''
                break_point = 0
                for j in range(piece1):
                    for i in range(piece2):
                        if temp[j,i] == 2:
                          if j < piece1 - 1:
                            for b in range(number):
                                m,n = pieces[b]
                
                                if ( m == j + 1 and n == i \
                               ) and rank[b] != 4 and rank[b] != 5 and rank[b]!=6:
                        
                                    maybe_edge_black_down += 1
                           
                                    if rank[b] == 8  or M[b]> -0.3:
                                        real_edge_black_down += 1
                            
                                    break
                          else:
                              maybe_edge_black_down = 0
                              break_point = 1
                              break
                    if break_point == 1:
                        break          
                                
                if maybe_edge_black_down > 1:                           
                    judge_down = np.float(real_edge_black_down) / maybe_edge_black_down
                    print "maybe_edge_black_down",maybe_edge_black_down
                    print "real_edge_black_down",real_edge_black_down
                    print "judge_down",judge_down
                    cv2.waitKey(0)
                
                maybe_edge_black_left = 0
                real_edge_black_left = 0
                judge_left = 0
                for j in range(piece1):
                    for i in range(piece2):
                        if temp[j,i] == 2:
                          
                            for b in range(number):
                                m,n = pieces[b]
                
                                if ( m == j  and n == i - 1 \
                               ) and rank[b] != 4 and rank[b] != 5 and rank[b]!=6:
                        
                                    maybe_edge_black_left += 1
                           
                                    if rank[b] == 8 or M[b]> -0.3:
                                        real_edge_black_left += 1
                            
                                    break
                         
                            
                                
                if maybe_edge_black_left > 1:                           
                    judge_left = np.float(real_edge_black_left) / maybe_edge_black_left
                    print "maybe_edge_black_left",maybe_edge_black_left
                    print "real_edge_black_left",real_edge_black_left
                    print "judge_left",judge_left
                    cv2.waitKey(0)

                maybe_edge_black_right = 0
                real_edge_black_right = 0
                judge_right = 0
                for j in range(piece1):
                    for i in range(piece2):
                        if temp[j,i] == 2:
                          
                            for b in range(number):
                                m,n = pieces[b]
                
                                if ( m == j  and n == i + 1 \
                               ) and rank[b] != 4 and rank[b] != 5 and rank[b]!=6:
                        
                                    maybe_edge_black_right += 1
                           
                                    if rank[b] == 8 or M[b]> -0.3:
                                        real_edge_black_right += 1
                            
                                    break
                         
                            
                                
                if maybe_edge_black_right > 1:                           
                    judge_right = np.float(real_edge_black_right) / maybe_edge_black_right
                    print "maybe_edge_black_right",maybe_edge_black_right
                    print "real_edge_black_right",real_edge_black_right
                    print "judge_right",judge_right
                    cv2.waitKey(0)

                
                for j in range(piece1):
                  for i in range(piece2):
                    if temp[j,i] == 2: 
                    
                
                        if judge_up >= 0.65 or judge_down >= 0.65 or judge_left >= 0.65 or judge_right >= 0.65:  
                            print S[a]
                            print "4级"
                            
                            temp[j,i] = 3
                            cv2.waitKey(0)
                   
                
                        else:
                            temp[j,i] = 0
                            rank[valid[j,i,7]] = 0
            else:
               for u in range(piece1):
                for v in range(piece2):
                    if temp[u,v] == 1 and valid[u,v,3] != -1:
                        print"u,v",u,v
                        print"valid[u,v,7]",valid[u,v,7]
                        print"rank[valid[u,v,7]]",rank[valid[u,v,7]]
                        cv2.waitKey(0)
                        temp[u,v] = 0
                        rank[valid[u,v,7]] = 0
               temp[J,I] = 0
               rank[valid[J,I,7]] = 0
            time = 0    
###################################
'''
maybe_edge_black_up = 0
real_edge_black_up = 0
judge_up = 0
for a in range(number):
        j,i = pieces[a]
        if rank[a] == 4:
            for b in range(number):
                m,n = pieces[b]
                
                if ( m == j - 1 and n == i\
                   #or m == j  and n == i - 1\
                   #or m == j  and n == i + 1\
                   #or m == j + 1 and n == i
                   ) and rank[b] != 4 and rank[b] != 5 and rank[b]!=6:
                        
                        maybe_edge_black_up += 1
                           
                        if S[b] > 1 or M[b]>0:
                            real_edge_black_up += 1
                            
                        break
                    
if maybe_edge_black_up != 0:                           
    judge_up = np.float(real_edge_black_up) / maybe_edge_black_up
    print "maybe_edge_black_up",maybe_edge_black_up
    print "real_edge_black_up",real_edge_black_up
    print "judge_up",judge_up
    cv2.waitKey(0)

maybe_edge_black_down = 0
real_edge_black_down = 0
judge_down = 0
for a in range(number):
        j,i = pieces[a]
        if rank[a] == 4:
            
            for b in range(number):
                m,n = pieces[b]
                
                if ( m == j + 1 and n == i\
                   #or m == j  and n == i - 1\
                   #or m == j  and n == i + 1\
                   #or m == j + 1 and n == i
                   ) and rank[b] != 4 and rank[b] != 5 and rank[b]!=6:
                        
                        maybe_edge_black_down += 1
                            
                        if S[b] > 1  or M[b]>0:
                            real_edge_black_down += 1
                        break
if maybe_edge_black_down != 0:                           
    judge_down = np.float(real_edge_black_down) / maybe_edge_black_down
    print "maybe_edge_black_down",maybe_edge_black_down
    print "real_edge_black_down",real_edge_black_down
    print "judge_down",judge_down
    cv2.waitKey(0)

   
for a in range(number):
            j,i = pieces[a]
            if rank[a] == 4:
                
                if judge_up >= 0.6 or judge_down >= 0.6:  
                    print S[a]
                    print "4级"
                    rank[a] = 4
                    cv2.waitKey(0)
                   
                
                else:
                    rank[a] = 0
'''            
               
for a in range(number):
    if rank[a] == 8:
            rank [a] = 0

for a in range(number):
    if M[a] < -1:
        j,i = pieces[a]
        print "j,i",j,i
        print "C",C[a]
        print "S",S[a]
'''
for a in range(number):
    j,i = pieces[a]
    if M[a] < -1 and C[a] > 0.5 and S[a]> 0.5  and rank[a] == 0:
        print "j,i",j,i
        print "C",C[a]
        print "S",S[a]
        rank[a] = 4
        print "第2种4级"
        cv2.rectangle(img2,((i) * stepx,(j ) * stepy),\
                            ((i + 1) * stepx - 1,(j + 1) * stepy - 1),(255,255,255),1)
'''
for a in range(number):
    j,i = pieces[a]
    if rank[a] == 6:
        cv2.rectangle(img2,((i) * stepx,(j ) * stepy),\
                            ((i + 1) * stepx - 1,(j + 1) * stepy - 1),(0,0,255),1)
    if rank[a] == 5:
        cv2.rectangle(img2,((i) * stepx,(j ) * stepy),\
                            ((i + 1) * stepx - 1,(j + 1) * stepy - 1),(255,0,255),1)
    if rank[a] == 4:
        cv2.rectangle(img2,((i) * stepx,(j ) * stepy),\
                            ((i + 1) * stepx - 1,(j + 1) * stepy - 1),(255,0,0),1)                        


#fill
def fill(edge,point):
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
        X1 = point
        A = 1 - edge
        X2 = cv2.bitwise_and(cv2.dilate(X1,kernel,iterations = 1),A)

        while (X2 != X1).any():
            X1 = X2
            X2 = cv2.bitwise_and(cv2.dilate(X1,kernel,iterations = 1),A)


        return edge + X1


def count_4(array):
    if (rank == 4).any():
        return np.bincount(array)[4]
    else:
        return 0


print  "rank", rank
if (rank == 6).any():
    print "这是6级！"
    cv2.imshow("img2",img2)
    cv2.waitKey(0)
elif (rank == 5).any():
    print "这是5级！"
    cv2.imshow("img2",img2)
    cv2.waitKey(0)
elif count_4(rank) >= 6:
    cv2.imshow("img2",img2)
    cv2.waitKey(0)
    print "这是4级！"
else:
    rank = np.zeros((number),np.uint8) 
    '''
    mean_wrong = np.array([[ 117.4051625],[ 143.4328375]])
    mean_right = np.array([[ 117.8874    ],[ 145.67651429]])
    cov_wrong = np.array([[ 19.09419453 ,-25.93495621],[-25.93495621 , 40.36949382]])
    cov_right = np.array([[ 16.42427016, -22.00842461],[-22.00842461 , 36.7027956 ]])


           
    #混合高斯模型


    #wrong
    y,x, channel = ycgcr.shape
    like_wrong = np.zeros((y,x),np.float64)
    MM = mean_wrong
    MM = np.repeat(MM,x,1)
   
    CC = cov_wrong
 
    C_1 = np.linalg.inv(CC)
    I = np.identity(x)                 
    cgcr = np.zeros((2,x),np.uint8)
    for j in range(y):
        cgcr[0,:] = ycgcr[j,:,1]
        cgcr[1,:] = ycgcr[j,:,2]
        like_wrong[j] = np.exp(-0.5 * np.sum(np.dot(np.dot(np.transpose(cgcr - MM),C_1),cgcr - MM) * I ,0))  


    #right
    y,x, channel = ycgcr.shape
    like_right = np.zeros((y,x),np.float64)
    MM = mean_right
    MM = np.repeat(MM,x,1)
    

    CC = cov_right
    C_1 = np.linalg.inv(CC)
    I = np.identity(x)                 
    cgcr = np.zeros((2,x),np.uint8)
    for j in range(y):
        cgcr[0,:] = ycgcr[j,:,1]
        cgcr[1,:] = ycgcr[j,:,2]
        like_right[j] = np.exp(-0.5 * np.sum(np.dot(np.dot(np.transpose(cgcr - MM),C_1),cgcr - MM) * I ,0))


    like_temp = like_wrong - like_right
    '''
    print "number",number
    cv2.waitKey(0)
    for a in range(number):
        j,i = pieces[a]
        print"j,i",j,i
        print "C",C[a]
        print "M",M[a]
        print "contrasts",contrasts[a]
        print "S",S[a]
        print "E",E[a]
        print "entropys",entropys[a]
        #print "ASMs",ASMs[a]
        if rank[a] == 0:
            if M[a] < -1 and C[a]> 1 and contrasts[a] > 0.15:#可以增加标准差限制
            #灰度值很低，但是也需要颜色位置信息辅助，否则阴影和黑斑分不开
                             
                black = 0
                for m in range(j*stepy, (j+1) * stepy):
                    for n in range(i*stepx, (i+1) * stepx):
                        if like_temp[m,n] > 0:
                                   black += 1
                                   
                #if black > stepy * stepx / 3 and black < 2 * stepy * stepx / 3:
                if black > stepy * stepx / 2:
                              
                        print j,i
                        #print a
                        
                        print "maybe 4级"
                        print "contrasts",np.average(contrasts[a])
                        print "standards",standards[a]
                        print "C",C[a]
                        print "S",S[a]
                        
                        
                        rank[a] = 4
                        '''
                        cv2.rectangle(img2,((i) * stepx,(j ) * stepy),\
                                       ((i + 1) * stepx - 1,(j + 1) * stepy - 1),(255,0,0),1)
                        '''
                        cv2.waitKey(0)
                        
                       
    width_thred = 2

    checked = np.zeros((number),np.uint8)

    for a in range(number):
       j,i = pieces[a] 
       if rank[a] == 4  and checked[a] == 0:
           team = np.array([a])
           checked[a] = 1
           count = 1
           b = a + 1
           if b < number:
               m,n = pieces[b]
               while m == j  and n == i + 1 and rank[b] == 4:
                   checked[b] = 1
                   count += 1
                   team = np.append(team,[b])
                   j = m
                   i = n
                   b += 1
                   if b>= number:
                   
                       break;
                   else:
                       m,n = pieces[b]
           print count
           cv2.waitKey(0)
           if count < width_thred:
               team_length = np.alen(team)
               for i in range (team_length):
                   rank[team[i]] = 0


    if (rank == 4).any():
        print "这是4级！"
        for a in range(number):
            j,i = pieces[a]
            if rank[a] == 4:
                cv2.rectangle(img2,((i) * stepx,(j ) * stepy),\
                            ((i + 1) * stepx - 1,(j + 1) * stepy - 1),(255,0,0),1)
                 
        cv2.imshow("img2",img2)
        cv2.waitKey(0)         
                 
    else:
                   
        gray5 =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #对output2缩小，去除可能存在背景的边缘部分
        ret,output2 = cv2.threshold(output2,0,1,cv2.THRESH_BINARY)
        kernel_size = stepx / 3 * 2 + 1
        kernel = np.ones((2 * kernel_size + 1,kernel_size),np.uint8)
        output2 = cv2.erode(output2,kernel,iterations = 1)
        ret,output2 = cv2.threshold(output2,0,255,cv2.THRESH_BINARY)
        mask_output2 = output2.copy()
        
        gray5 =  cv2.bitwise_and(gray5,mask_output2)
        cv2.imshow("gray5",gray5)
        cv2.waitKey(0)
   
        gray6 = gray5.copy()
        gray5 = cv2.medianBlur(gray5,7)
        
        kernel_size = 11 + distance / 3 * 6
        print "kernel_size",kernel_size
        dst3 = cv2.adaptiveThreshold(gray5,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        cv2.THRESH_BINARY,kernel_size,3)
        dst3 = 255 - dst3

        dst4 = cv2.adaptiveThreshold(gray5,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        cv2.THRESH_BINARY,kernel_size,-3)

        if max_area / Max <= 250:
            meanKernel = 7
        else:
            meanKernel = 9
        print meanKernel
        dst3 = cv2.medianBlur(dst3,meanKernel)
        dst4 = cv2.medianBlur(dst4,meanKernel)


        cv2.waitKey(0)


 



        dst3 = cv2.bitwise_and(dst3,mask_output2)

        ret,output3 = cv2.threshold(output2,0,1,cv2.THRESH_BINARY)
        erodeKernel = kernel_size
        kernel = np.ones((erodeKernel+4,erodeKernel+4),np.uint8)
        output3 = cv2.erode(output3,kernel,iterations = 1)
        ret,output3 = cv2.threshold(output3,0,255,cv2.THRESH_BINARY)

        mask_output3 = output3.copy()
        dst4 = cv2.bitwise_and(dst4,mask_output3)



        cv2.imshow("dst3",dst3)
        cv2.imshow("dst4",dst4)
        cv2.waitKey(0)
        '''
        dst7 = cv2.bitwise_or(dst3,dst4)
        cv2.imshow("dst7",dst7)
        '''
        ret,dst3 = cv2.threshold(dst3,0,1,cv2.THRESH_BINARY)
        ret,dst4 = cv2.threshold(dst4,0,1,cv2.THRESH_BINARY)

        kernel = np.ones((kernel_size / 4 * 2 + 1 ,kernel_size / 4 * 2 + 1),np.uint8)
        dst5 = cv2.dilate(dst3,kernel,iterations = 1)
        dst6 = cv2.dilate(dst4,kernel,iterations = 1)

        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
        dst5_dilate = cv2.dilate(dst5,kernel,iterations = 1)
        dst5_edge = dst5_dilate - dst5
        dst6_dilate = cv2.dilate(dst6,kernel,iterations = 1)
        dst6_edge = dst6_dilate - dst6
    

        dst7 = cv2.bitwise_and(dst5,dst6)
        #cv2.imshow("dst7",dst7)
        cv2.waitKey(0)

      

   


        dst_sheld = cv2.bitwise_and(fill(dst5_edge, dst7) , dst3)
        dst_light = cv2.bitwise_and(fill(dst6_edge, dst7) , dst4)

        dst_final = cv2.bitwise_or(dst_sheld,dst_light)

        ret,dst_final_temp = cv2.threshold(dst_final,0,255,cv2.THRESH_BINARY)

        cv2.imshow("dst_final",dst_final_temp)
        cv2.waitKey(0)

   

        #静脉曲张面积
        square = np.sum(dst_final)
        ratio = np.float64(Min) / 100
        if leg == 1:
            judge = np.float64(square) /  max_area
        else:
            judge = np.float64(square) /  (max_leg[1,0] + max_leg[1,1])
        print square,max_area,judge
        cv2.waitKey(0)

        #粗的静脉太少，则是0或1级

        #因为使用面积而不是长度作为判断，静脉图中的小点不会对结果造成很大的影响

        if judge < 0.001:#可以尝试canny算法
            
            contrast_sort = contrasts[np.argsort(contrasts)]
            contrast_sort = contrast_sort[::-1]
            print "contrast_sort",contrast_sort
            C_max = np.max(C)
            
            print "C_max",C_max
            cv2.waitKey(0)
            
            if  contrast_sort[2] > 0.159:
                print "这是1级"
            else:
                print "这是0级"

            
            
        elif judge >= 0.001 and judge < 0.03:
            print "这是2级"
        else:
            print "这是3级"
    
    



    

cv2.waitKey(0)
cv2.waitKey(0)
cv2.waitKey(0)
cv2.waitKey(0)
cv2.destroyAllWindows()
