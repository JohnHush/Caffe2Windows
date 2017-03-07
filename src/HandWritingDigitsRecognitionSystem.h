#include <opencv2/opencv.hpp>
#include <vector>
using std::vector;

//void initPredictor( int BLOCK_SIZE=201 , double OFFSET=20 );
//void deletePredictor();
//int looksLikeNumber( IplImage * imgSrc   , float & confidence , float red_pts_prec = 0.1 );

extern __declspec(dllexport) void initPredictor( int BLOCK_SIZE=201 , double OFFSET=20 );
extern __declspec(dllexport) void deletePredictor( );
extern __declspec(dllexport) int looksLikeNumber( IplImage * imgSrc   , float & confidence , float red_pts_prec = 0.1 );
