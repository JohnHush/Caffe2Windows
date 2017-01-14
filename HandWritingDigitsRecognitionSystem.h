#include <opencv2/opencv.hpp>
#include <vector>
using std::vector;

//extern __declspec(dllexport) void initPredictor( int BLOCK_SIZE=201 , double OFFSET=20 , int epsilon = 20 , int iteration= 10 , vector<IplImage *> & imgs = vector<IplImage *>(0) );
//extern __declspec(dllexport) void deletePredictor( );
//extern __declspec(dllexport) int looksLikeNumber( IplImage * imgSrc , float if_less_than_then_its_blank = 0.1 , float keep_at_least_area = 0.8 );

void initPredictor( int BLOCK_SIZE=201 , double OFFSET=20 , int epsilon = 20 , int iteration= 10 , vector<IplImage *> & imgs = vector<IplImage *>(0) );
void deletePredictor( );
int looksLikeNumber( IplImage * imgSrc , float if_less_than_then_its_blank = 0.1 , float keep_at_least_area = 0.8 );