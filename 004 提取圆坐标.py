import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

####################
# 读取图片
####################

def process(file):

    # file = '154699_3_iris_f.png'
    if file == 'Thumbs.db': return # 直接结束
    
    img = mpimg.imread( path + file )
    '''
    print('img.shape =', img.shape)
    print()
    print('=== range ===')
    print(np.min(img[:,:,0]), np.max(img[:,:,0])) # 这图片是float 0-1
    print(np.min(img[:,:,1]), np.max(img[:,:,1])) # 这图片是float 0-1
    print(np.min(img[:,:,2]), np.max(img[:,:,2])) # 这图片是float 0-1
    print()
    '''
    # plt.imshow(img[:,:,0]) # 内圆
    # plt.imshow(img[:,:,2]) # 轮廓
    # plt.show()

    ####################
    # 获取索引
    ####################

    area = np.sum(img[:,:,0])
    r = int(np.sqrt( area / np.pi )) # 获取半径

    index = np.argwhere(img[:,:,0]==1)
    x = int(np.mean(index[:,0])) # 获取圆心坐标
    y = int(np.mean(index[:,1]))

    mask = img[:,:,2]
    R = 0 # 虹膜最大半径
    for i in range(200, mask.shape[0]-200):
        for j in range(200, mask.shape[1]-200):
            if mask[i, j] == 0: continue # 黑色则跳过
            # 计算最大外圈距离（半径）
            dist = np.linalg.norm( np.array([i, j]) - np.array([x, y]) )
            dist = int(dist)
            if dist > R: R = dist # 找到最大
    '''
    cv2.circle(img, (y, x), r, (0,255,0), 2)
    cv2.circle(img, (y, x), R, (0,255,0), 2)

    cv2.imshow('detected circles', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    '''
    print(file, end=',', file=fw)
    print(x, end=',', file=fw)
    print(y, end=',', file=fw)
    print(r, end=',', file=fw)
    print(R, end=',', file=fw)
        '''
    content = file+','+str(x-200)+','+str(y-200)+','+str(r)+','+str(R)+'\n'
    return content

####################
# 写入
# 使用回调进行同步
####################

def callback(content):
    with open( parameter + '_circle.csv', 'a' ) as fw:
        if content: # 最后返回none
            fw.write(content) # 回调锁住了，不需要快

####################
# 多进程
####################

from multiprocessing import Pool

import os
parameter = 'left'
path = '003 data mask marked/' + parameter + '/'

if __name__ == '__main__':

    fw = open( parameter + '_circle.csv', 'w' ) # 删除之前内容
    print('name,x,y,r,R', file=fw)
    fw.close()

    files = os.listdir(path)

    pool = Pool() # python来决定 processes=16/24/32 没区别，python不会用爆cache
    for file in files:
        pool.apply_async( func = process,
                          args = (file,),
                          callback = callback )
    pool.close()
    pool.join()
