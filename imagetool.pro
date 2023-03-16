#-------------------------------------------------
#
# command line Image tool for RC.
# Features:
#  -Improves contrast on nDNA images
#  -TODO
#
#-------------------------------------------------
CONFIG -= qt
CONFIG += console 

TARGET = opencvtest
TEMPLATE = vcapp



SOURCES += main.cpp\




INCLUDEPATH += . \
		"../Libraries/3rdparty/opencv/include/opencv" \
		"../Libraries/3rdparty/opencv/include/" \

win32 {
	CONFIG(debug, debug|release) {
		LIBS += \
		../Libraries/3rdparty/opencv/lib/opencv_contrib240d.lib \
		../Libraries/3rdparty/opencv/lib/opencv_core240d.lib \
		../Libraries/3rdparty/opencv/lib/opencv_features2d240d.lib \
		../Libraries/3rdparty/opencv/lib/opencv_highgui240d.lib \
		../Libraries/3rdparty/opencv/lib/opencv_imgproc240d.lib \
		../Libraries/3rdparty/opencv/lib/opencv_ml240d.lib \
		../Libraries/3rdparty/opencv/lib/opencv_objdetect240d.lib 
		DESTDIR = ../outputd
	} else {
		LIBS += \
		../Libraries/3rdparty/opencv/lib/opencv_contrib240.lib \
		../Libraries/3rdparty/opencv/lib/opencv_core240.lib \
		../Libraries/3rdparty/opencv/lib/opencv_features2d240.lib \
		../Libraries/3rdparty/opencv/lib/opencv_highgui240.lib \
		../Libraries/3rdparty/opencv/lib/opencv_imgproc240.lib \
		../Libraries/3rdparty/opencv/lib/opencv_ml240.lib \
		../Libraries/3rdparty/opencv/lib/opencv_objdetect240.lib 
		DESTDIR = ../output
	}
}