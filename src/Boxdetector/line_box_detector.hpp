#ifndef __JohnHush_LINE_BOX_DETECTOR_H_
#define __JohnHush_LINE_BOX_DETECTOR_H_

#include "../boxDetector.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include "../util.hpp"

using std::pair;

class LineBoxDetector : public BoxDetector
{
	protected:
		float spaceGap_;
		/**
		 * filter out the lines within the gap in both direction
		 * which means we only need some lines don't gather together
		 */
	public:
		explicit LineBoxDetector( IplImage * imgSrc , float space_gap ): 
			BoxDetector( imgSrc ), spaceGap_(space_gap){}
		virtual void detectBox();
		virtual ~LineBoxDetector(){};
		void showOnImage() const ;
		inline static bool sortT( const pair<int , int > & a , const pair<int , int> & b)
		{
			return (abs(a.first) + abs(a.second)) > (abs(b.first) + abs(b.second));
		}
};

#endif
