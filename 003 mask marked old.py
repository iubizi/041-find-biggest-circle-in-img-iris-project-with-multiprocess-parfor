# import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

parameter = 'left'

####################
# 计算距离
####################

def calc(i, j, mask):
    '''
    # 暴力算法不可取
    dist = 10000
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x, y] == 1: # 测算到1的点的最小距离
                temp = np.linalg.norm( np.array([x, y]) -
                                       np.array([i, j]) )
                if temp < dist: dist = temp # 找最小值
    '''
    dist = 10000
    # 应该是最快的取值
    step = int(np.sqrt(np.sqrt(mask.shape[0]*mask.shape[1]))) # 每次扩展值
    if step > 15: step = 15 # 避免太大
    flag = 0 # 是否出现，出现就停止

    for k in range(100):
        # 限定范围，不要超了
        x_min = i-k*step if i-k*step >= 0 else 0
        x_max = i+k*step if i+k*step <= mask.shape[0] else mask.shape[0]
        # print(x_min, x_max)
        for x in range(x_min, x_max):
            # 限定范围，不要超了
            y_min = j-k*step if j-k*step >= 0 else 0
            y_max = j+k*step if j+k*step <= mask.shape[1] else mask.shape[1]
            # print(y_min, y_max)
            for y in range(y_min, y_max):
                if mask[x, y] == 1: # 测算到1的点的最小距离
                    temp = np.linalg.norm( np.array([x, y]) -
                                           np.array([i, j]) )
                    # print(temp)
                    if temp < dist:
                        dist = temp # 找最小值
                        flag = 1
        
        if flag: break # 标志位说明找到，结束
    
    return dist

####################
# 读取图片
####################

def process(file):
    # file = '20992_3_iris_f.png'
    if file == 'Thumbs.db': return # 直接结束

    img = mpimg.imread( '002 data mask/' + parameter + '/' + file )
    print('img.shape =', img.shape)
    print()
    print('=== range ===')
    print(np.min(img[:,:,0]), np.max(img[:,:,0])) # 这图片是float 0-1
    print(np.min(img[:,:,1]), np.max(img[:,:,1])) # 这图片是float 0-1
    print(np.min(img[:,:,2]), np.max(img[:,:,2])) # 这图片是float 0-1
    print()

    # 获取模板
    mask_all = img[:,:,2] # 存储于蓝色维度
    print('=== mask_all ===')
    print('mask_all.shape =', mask_all.shape)
    print(np.min(mask_all), np.max(mask_all))
    print()

    resize = 2 # 缩小2-4倍以便于处理，太小会导致不精确
    mask_shape = [ mask_all.shape[0]//resize,
                   mask_all.shape[1]//resize ]

    mask = np.zeros(mask_shape)
    for i in range(mask_shape[0]):
        for j in range(mask_shape[1]):
            mask[i, j] = mask_all[i*resize, j*resize]

    print('=== mask ===')
    print('mask.shape =', mask.shape)
    print(np.min(mask), np.max(mask))
    print()

    # input() # 前面的都对

    ####################
    # 取点
    ####################

    max_i, max_j, max_r = 0, 0, 0 # 找最大圆

    for i in range(int(mask.shape[0]*0.3), int(mask.shape[0]*0.7)):
        print(i, end=' ') # 查看进度
        for j in range(int(mask.shape[1]*0.3), int(mask.shape[1]*0.7)):
            # print(j, end=' ') # 查看进度
            if mask[i, j] == 0:
               temp = calc(i, j, mask) # 测算距离
               # print(temp)
               if temp > max_r: # 找最大值
                   max_r = int(temp)
                   max_i, max_j = i, j

    ####################
    # 可视化 保存
    ####################

    max_i *= resize
    max_j *= resize
    max_r *= resize

    # 绘制实心圆
    print()
    print()
    print('=== circle size ===')
    print(max_i, max_j, max_r)

    img[:,:,0] = 0 # 清空这个图层，本图层只有标记圆 # 红色
    img[:,:,1] *= 255 # float转int
    img[:,:,2] *= 255 # float转int

    try: # 有些图是坏的
        cv2.circle(img, (max_j, max_i), max_r, (255,0,0), -1) # 和蓝色重合能看出来
    except:
        pass
    '''
    cv2.imshow('detected circles', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    cv2.imwrite( '003 data mask marked/' + parameter + '/' + file,
                 cv2.cvtColor(img, cv2.COLOR_RGB2BGR) )

####################
# 多进程
####################

import os

path = '002 data mask/' + parameter
files = os.listdir(path)

from multiprocessing import Pool

if __name__ == '__main__':
    
    pool = Pool(16) # python来决定 processes=16/24/32 没区别，python不会用爆cache
    pool.map(process, files)
    pool.close()
    pool.join()
