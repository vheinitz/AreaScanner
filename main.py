
"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np
from datetime import datetime

exposure = 0.0

cap = cv.VideoCapture(1, cv.CAP_DSHOW )
cap.set(cv.CAP_PROP_FRAME_WIDTH ,2560)
cap.set(cv.CAP_PROP_FRAME_HEIGHT ,1944)
cap.set(cv.CAP_PROP_AUTO_EXPOSURE ,0.01)
ret = cap.set(cv.CAP_PROP_EXPOSURE , -2.0)
#ret = cap.set(cv.CAP_PROP_GAIN , 1)
#ret = cap.set(cv.CAP_PROP_MODE, 3 )


if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, frame = cap.read()
print("WxH:", frame.shape, ret )

while True:
    # Capture frame-by-frame
    ret, src = cap.read()
    frame=0
    frame = cv.pyrDown(src)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = frame #cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', gray)
    k = cv.waitKey(1)
    if k == ord('q'):
        break

    elif k == ord('s'):
        cv.imwrite("c:/temp/cv_%d_%d_%d.png"%(datetime.now().hour,datetime.now().minute,datetime.now().second),src)
    elif k == ord('a'):
        exposure+=1
        cap.set(cv.CAP_PROP_EXPOSURE, exposure)

    elif k == ord('y'):
        exposure-=1
        cap.set(cv.CAP_PROP_EXPOSURE, exposure)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
"""
def main(argv):
    default_file = 'c:/tmp/lc.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    cv.imshow("Source", src)
    #src = cv.pyrDown(src, src)
    #src = cv.pyrDown(src, src)
    #src = cv.pyrDown(src, src)

    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    canny = cv.Canny(src, 50, 200, None, 3)
    cv.imshow("Canny", canny)

    kernel = np.ones((5, 5), np.uint8)
    dilate = cv.dilate(canny, kernel, iterations=1)
    cv.imshow("dilate", dilate)

    dist = cv.distanceTransform(dilate, cv.DIST_L2, 3)

    dist = dist.astype('uint8')
    (_, dist) = cv.threshold(dist, 5, 255, cv.THRESH_BINARY );
    cv.normalize(dist, dist, 0, 255, cv.NORM_MINMAX);
    #
    cv.imshow("dist", dist)

    cv.waitKey(10)

    (contours, _) = cv.findContours(cdst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        # cv.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
        area = cv.contourArea(contour)

        #if area < 8000:
        cv.drawContours(src, [contour], 0, (0, 0, 0), -1)



    cv.imshow("dil", dil)
    cv.waitKey(10)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 10, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 1, cv.LINE_AA)

    linesP = cv.HoughLinesP(cdst, 1, np.pi / 180, 50, None, 50, 10)

    (contours,_) = cv.findContours(cdst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        #cv.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
        area = cv.contourArea(contour)

        (x, y, w, h) = cv.boundingRect(contour)
        if max(w,h)/min(w,h) < 1.2 and area > 40000:
            tmp = src.copy()

            (_, empty) = cv.threshold(tmp, 250, 0, 0)
            cv.drawContours(empty, [contour], 0, (255, 255, 255), -1)

            #cv.waitKey(1000)
            (_,mask) = cv.threshold(tmp,254,255,0)
            #mask =cv.bitwise_not(mask)
            #cv.bitwise_and(tmp, mask, tmp)
            tmp = cv.bitwise_or(tmp, tmp, mask=mask)

            tmp = cv.Canny(tmp, 50, 200, None, 3)
            (lcc_contours, _) = cv.findContours(tmp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            for lc in lcc_contours:

                cv.drawContours(empty, [lc], 0, (0, 0, 0), -1)

            #tmp = cv.pyrDown(tmp)
            tmp = cv.pyrDown(tmp)
            tmp = cv.pyrDown(tmp)
            #empty = cv.pyrDown(empty)
            #empty = cv.pyrDown(empty)
            cv.imshow("tmp", tmp)
            cv.imshow("empty", empty)

            cv.waitKey(500)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv.LINE_AA)

    #src = cv.pyrDown(src)
    #src = cv.pyrDown(src)
    #cv.imshow("Source", src)
    #cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    #cdstP = cv.pyrDown(cdstP)
    #cdstP = cv.pyrDown(cdstP)
    #cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv.waitKey()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
"""