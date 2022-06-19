# 041-find-biggest-circle-in-img-iris-project-with-multiprocess-parfor

041 find biggest circle in img (iris project with multiprocess parfor)

## 003 mask marked.py

问题描述：在一张黑色背景的虹膜照片中找到瞳孔，其中瞳孔是黑色的，虹膜占据红色绿色通道，但是蓝色通道是虹膜蒙板，使用半径最大的圆标注出来（红色）

使用多进程，首先找到所有黑色的点，然后测算其到白色点的最小距离作为其可用半径，找出所有黑色点中最大的可用半径，之后保存xy坐标和半径，绘图

为了让标注都在图片里面，所以在最终标注之前扩边200，这样所有的圆全部都在图片里面（old改进）
