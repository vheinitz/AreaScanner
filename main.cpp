#include <cv.h>
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <list>
#include <vector>
#include <cmath>


namespace geometry
{

	class Point{
		Point(float x, float y);
		//Point(QPoint);
		Point(cv::Point);
		Point(cv::Point2f);
		operator cv::Point();
		//operator QPoint();
		operator cv::Point2f();
	};
	
};


class VideoSim
{
public:
	std::string _fn;
	cv::Size _wsize;
	cv::Mat _mat;
	int _x;
	int _y;

	VideoSim( cv::Size wsize, std::string fn ):_wsize(wsize), _fn(fn)
	{		
		_mat = cv::imread(_fn); 
		_x = _mat.cols / 2 - _wsize.width;
		_y = _mat.rows / 2 - _wsize.height;
	}

	cv::Mat read()
	{

		return _mat(cv::Rect(_x, _y, _wsize.width, _wsize.height)).clone();
	}

	void changeY( int dif)
	{
		if ( _y+dif < 0 )
		{
			return;
		}
		if ( _y+dif+_wsize.height >= _mat.rows )
		{
			return;
		}

		_y += dif;
	}

	void changeX( int dif)
	{
		if ( _x+dif < 0 )
		{
			return;
		}
		if ( _x+dif+_wsize.width >= _mat.cols )
		{
			return;
		}
		_x += dif;
	}

};



////////////////////////////////
class SceneBuilder
{
	std::vector<cv::KeyPoint> keypoints_prev, keypoints_cur;
	cv::Mat descriptors_prev, descriptors_cur;

	//cv::MSER detector;
	//cv::StarDetector detector;
	cv::GFTTDetector detector;
	//cv::OrbFeatureDetector detector(500,1.2,8,32,0,2,cv::ORB::HARRIS_SCORE,31);
	//cv::FastFeatureDetector detector;
	//cv::DenseFeatureDetector detector;

	cv::BriefDescriptorExtractor extractor;

	cv::BFMatcher matcher;
	std::vector<cv::DMatch> matches;
	cv::Mat img_matches;

	int sceneX;
	int sceneY;
	cv::Size _wsize;

	cv::Mat curFrame;
	cv::Mat prevFrame;
	cv::Mat resultScene;


public:
	SceneBuilder() :sceneX(0),sceneY(0),matcher(cv::NORM_L1)
	{
	}

	void push( cv::Mat frame )
	{
		curFrame = frame;		

		if ( prevFrame.empty() )
		{
			resultScene = curFrame.clone();
			prevFrame = curFrame.clone();
			_wsize = curFrame.size();
			return;
		}

		if ( _wsize != curFrame.size() )
		{
			reset();
			return;
		}

		detector.detect(curFrame, keypoints_cur );
		extractor.compute(curFrame, keypoints_cur, descriptors_cur );

		detector.detect(prevFrame, keypoints_prev);
		extractor.compute(prevFrame, keypoints_prev, descriptors_prev);

		
		if(!descriptors_cur.empty() && !descriptors_prev.empty()) 
		{
			matcher.match (descriptors_cur, descriptors_prev, matches);

			double max_dist = 0; double min_dist = 100;

			double distAvg = 0;
			std::list<int> ldistX;
			std::list<int> ldistY;
			for( int i = 0; i < descriptors_cur.rows; i++)
			{ 
				double dist = matches[i].distance;
				if( dist < min_dist ) min_dist = dist;
				if( dist > max_dist ) max_dist = dist;
				distAvg += dist;
			}

			distAvg /=descriptors_cur.rows;
	
			std::vector< cv::DMatch >good_matches;

			double xDiff=0;
			double yDiff=0;
			for( int i = 0; i < descriptors_cur.rows; i++ )
			{ 
				if( matches[i].distance <= (distAvg) )
				{ 
					good_matches.push_back( matches[i]);
					//xDiff +=  keypoints_cur[ matches[i].queryIdx ].pt.x - keypoints_prev[ matches[i].trainIdx ].pt.x;
					//yDiff +=  keypoints_cur[ matches[i].queryIdx ].pt.y - keypoints_prev[ matches[i].trainIdx ].pt.y;

					ldistX.push_back(keypoints_cur[ matches[i].queryIdx ].pt.x - keypoints_prev[ matches[i].trainIdx ].pt.x);
					ldistY.push_back(keypoints_cur[ matches[i].queryIdx ].pt.y - keypoints_prev[ matches[i].trainIdx ].pt.y);

				}
			}
			ldistX.sort();
			ldistY.sort();
			if ( ldistX.size() >= 5 )
			{
				std::list<int>::iterator it = ldistX.begin();
				std::advance(it, ldistX.size()/2);
				xDiff = *it;
			}

			if ( ldistY.size() >= 5 )
			{
				std::list<int>::iterator it = ldistY.begin();
				std::advance(it, ldistY.size()/2);
				yDiff = *it;
			}

			

			/*if ( good_matches.size() )
			{
				xDiff /= good_matches.size();
				yDiff /= good_matches.size();
			}*/

			int movedX = std::floor(xDiff + 0.5);
			int movedY = std::floor(yDiff + 0.5);

			cv::imshow( "CurFrame", curFrame );
			cv::imshow( "PrevFrame", prevFrame );

			if ( movedX || movedY )
			{
				sceneX -= movedX;
				sceneY -= movedY;
				
				int sceneXOld = 0;
				int sceneYOld = 0;

				int newWidth=resultScene.cols;
				int newHeight=resultScene.rows;

				if ( sceneX < 0 )
				{
					newWidth += std::abs(sceneX);
					sceneXOld = std::abs(sceneX);
					sceneX=0;
				}
				else
				{
					newWidth = std::max( sceneX + _wsize.width, resultScene.cols ); 
				}

				if ( sceneY < 0 )
				{
					newHeight += std::abs(sceneY);
					sceneYOld = std::abs(sceneY);
					sceneY=0;
				}
				else
				{
					newHeight = std::max( sceneY + _wsize.height, resultScene.rows ); 
				}

				if ( newWidth > resultScene.cols || newHeight > resultScene.rows )
				{
					cv::Mat tmpNewScene = cv::Mat::zeros(newHeight, newWidth, resultScene.type());		
					if ( sceneX < sceneXOld || sceneY < sceneYOld )
					{						
						resultScene.copyTo(tmpNewScene( cv::Rect(sceneXOld, sceneYOld, resultScene.cols, resultScene.rows) ));						
					}
					else
					{						
						resultScene.copyTo(tmpNewScene( cv::Rect(sceneXOld, sceneYOld, resultScene.cols, resultScene.rows) ));										
					}					
					resultScene = tmpNewScene;
				}
				curFrame.copyTo(resultScene( cv::Rect(sceneX, sceneY, _wsize.width,_wsize.height) ));

				prevFrame = curFrame.clone();


			}

			cv::imshow( "resultScene", resultScene );

			cv::Mat vd_kp1, vd_kp2, vd_kp3;

			cv::drawMatches( curFrame, keypoints_cur, prevFrame, keypoints_prev,
					good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1));

			cv::imshow( "Match", img_matches );
		}
	}

	cv::Mat result() 
	{
		return resultScene;
	}

	void reset() 
	{
		sceneX=0;
		sceneY=0;
		prevFrame = cv::Mat();
		resultScene = cv::Mat();
	}	
};
SceneBuilder scene;
////////////////////////////////


int main(int argc, char** argv)
{
	VideoSim cap( cv::Size(400,400), "C:/tmp/sim.png" );
	SceneBuilder scene;

	while (true)
	{

		scene.push (cap.read());
		
		//cv::imshow( "Frame", frames[curFrameIdx] );
		char k = cv::waitKey(10);
		if (k == 'q' )
			break;
		else if (k == '1' )
			cap.changeX(-10);
		else if (k == '2' )
			cap.changeX(+10);
		else if (k == '3' )
			cap.changeY(-10);
		else if (k == '4' )
			cap.changeY(+10);
		else if (k == 'r' )
		{
			scene.reset();
		}
		
		
		

		
	}


//for(;;) {
	/*
    cv::Mat frame = cv::imread("E:\\Projects\\Images\\2-134-2.bmp", 1);
    cv::Mat img_scene = cv::Mat(frame.size(), CV_8UC1);
    cv::cvtColor(frame, img_scene, cv::COLOR_RGB2GRAY);
    //frame.copyTo(img_scene);
    if( method == 0 ) { //-- ORB
        orb.detect(img_scene, keypoints_scene);
        orb.compute(img_scene, keypoints_scene, descriptors_scene);
    } else { //-- SURF
        detector.detect(img_scene, keypoints_scene);
        extractor.compute(img_scene, keypoints_scene, descriptors_scene);
    }

    //-- matching descriptor vectors using FLANN matcher
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    cv::Mat img_matches;
    if(!descriptors_object.empty() && !descriptors_scene.empty()) {
        matcher.match (descriptors_object, descriptors_scene, matches);

        double max_dist = 0; double min_dist = 100;

        //-- Quick calculation of max and min idstance between keypoints
        for( int i = 0; i < descriptors_object.rows; i++)
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        //printf("-- Max dist : %f \n", max_dist );
        //printf("-- Min dist : %f \n", min_dist );
        //-- Draw only good matches (i.e. whose distance is less than 3*min_dist)
        std::vector< cv::DMatch >good_matches;

        for( int i = 0; i < descriptors_object.rows; i++ )

        { if( matches[i].distance < (max_dist/1.6) )
            { good_matches.push_back( matches[i]); }
        }

        cv::drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, \
                good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
*/
	///////////////

	return 0;

}
