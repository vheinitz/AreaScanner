import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

tmpInputImages = []
#img1 = cv.imread('c:/tmp/lc.png', cv.IMREAD_GRAYSCALE)          # queryImage
simSlideArea = cv.imread('c:/tmp/gk.jpg', cv.IMREAD_GRAYSCALE) # trainImage
micAreaSize = ( 200, 200 )

x = 100
y = 100

curImg = simSlideArea[x:x+micAreaSize[0], y:y+micAreaSize[1]]        # queryImage
prevImg = curImg
resultImage = curImg
# Initiate SIFT detector
fdet = cv.SIFT_create()
fdet = cv.FastFeatureDetector_create()
bf = cv.BFMatcher()


resultSize = [curImg.shape[0], curImg.shape[1]]
roiPosition = [0,0]

while True:
    #center = curImg[int(micAreaSize[0]/4):int(micAreaSize[0]*.75),int(micAreaSize[1]/4):int((micAreaSize[1]*.75)]
    center = curImg[50:150,50:150]
    #kp1, des1 = fdet.detectAndCompute(center, None)
    #kp2, des2 = fdet.detectAndCompute(prevImg, None)
    kp1 = fdet.detect(center, None)
    kp2 = fdet.detect(prevImg, None)
    good = []
    if len(kp1)>0 and len(kp2)>0 and len(kp2) > len(kp1):
        matches = bf.knnMatch(des1,des2,k=2)
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append([m])
        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv.drawMatchesKnn(center, kp1, prevImg, kp2, good, None,
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # Apply ratio test
    prevImg = curImg

    dX = 0
    dY = 0
    dXa = []
    dYa = []
    for m in good:
        dXa.append(kp2[m[0].trainIdx].pt[0] - 50 - kp1[m[0].queryIdx].pt[0])
        dYa.append(kp2[m[0].trainIdx].pt[1] - 50 - kp1[m[0].queryIdx].pt[1])

    np.sort(dXa)
    np.sort(dYa)

    if (  len(good) > 5):
        dX = round(dXa[ round(len(dXa)/2)])
        dY = round(dYa[round(len(dYa) / 2)])

    if dX != 0:
        print(dXa)

    #print(int(dX), " ", int(dY))

    #cv.imshow("curImg", curImg)
    #cv.imshow("prevImg", prevImg)
    cv.imshow("Result", resultImage)

    cv.imshow("Out", img3)
    k = cv.waitKey(100)
    if k == ord('q'):
        break
    elif k == ord('1'):
        x -= 10
        if x < 10:
            x=10
    elif k == ord('2'):
        x += 10
        if x+micAreaSize[0]+10 >= simSlideArea.shape[0]:
            x -= simSlideArea.shape[0]- micAreaSize[0]-10

    elif k == ord('3'):
        y -= 10
        if y < 0:
            y=0
    elif k == ord('4'):
        y += 10
        if y+micAreaSize[1] >= simSlideArea.shape[1]:
            y -= 10

    roiPosition[0] += dX
    roiPosition[1] += dY

    if roiPosition[0] < 0:
        resultSize[0] += round(math.fabs(roiPosition[0]) )

        roiPosition[0] =0

    if roiPosition[1] < 0:
        resultSize[1] += round(math.fabs(roiPosition[1]))
        roiPosition[1] = 0
    if roiPosition[0] + micAreaSize[0] >= resultSize[0]:
        resultSize[0] += round(( roiPosition[0] + micAreaSize[0] ) - resultSize[0])
    if roiPosition[1] + micAreaSize[1] > resultSize[1]:
        resultSize[1] += round(( roiPosition[1] + micAreaSize[1] ) - resultSize[1])

    curImg = simSlideArea[y:y + micAreaSize[1], x:x + micAreaSize[0]]  # queryImage
    #tmp = np.zeros(resultSize, np.float32)
    tmp = np.zeros(( resultSize[1], resultSize[0]), np.uint8)

    tmp[0:resultImage.shape[0], 0:resultImage.shape[1]] = resultImage
    tmp[roiPosition[1]:roiPosition[1] + micAreaSize[1], roiPosition[0]:roiPosition[0] + micAreaSize[0]] = curImg
    resultImage = tmp


    if dX!=0 or dY!=0:
        print (dX, dY,  resultSize, roiPosition)


#plt.imshow(img3)
#plt.show()