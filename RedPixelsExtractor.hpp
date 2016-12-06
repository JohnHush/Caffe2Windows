#ifndef REDPIXELSEXTRACTOR_H
#define REDPIXELSEXTRACTOR_H

#include <vector>
#include <utility>
#include <cmath>

using namespace std;

class RedPixelsExtractor
/*
 * The Extractor can only use two features in the algorithm,
 * we recommend using feature1: (R + epsilon)/(G + epsilon) and feature2: (R + epsilon)/(B + epsilon)
 * and it only can output two class, red or not.
 * the program is designed specially for red pixels extraction
 * don't use it in any other circumstances
 */
{
	typedef struct mat2d
	{
		float a00;
		float a01;
		float a10;
		float a11;
	}mat2d;

	private:
		vector< pair<float , float> > features_;
		/*
		 * features input for learning, the features is filtered by cvAdaptiveThreshold
		 * users could use other features but the number should be fixed in two.
		 * because higher dimension matrix's inversion should be calculated in BLAS or 
		 * some other software, in this condition, to simplify the calculation we constrain
		 * the calculation in two dimension
		 */
		vector<float> phai_;
		/*
		 * the a prior probability of the class, used to compute posterior probability
		 * this value should be initialed by the user, and should be summed as 1.
		 * the value should be set compares to the number points ration of each class.
		 * i.e. if the red points and black points are nearly have the same number, then
		 * it should be set as (0.5, 0.5);
		 */
		vector<mat2d> sigma_;
		/*
		 * could be initialized as Identity matrix
		 */
		vector< pair<float , float> > exp_;
		/*
         * the expectation of the feature vector, 
         */
		vector< pair<float , float> > post_;
		/*
 		 * the posterior probability of every single feature data
 		 * computed by using phai_, exp_, sigma_ 
 		 * in Bayes theory
 		 */
		vector<float> delta_phai_;
		vector< pair<float , float> > delta_exp_;
		/*
 		 * recording the parameter changing in each iteration
 		 * we stop the iteration when every parameter's updating
 		 * is less than some threshold, maybe i.e. 2% 
 		 */

		void matrix_inversion( mat2d & a , mat2d & inverse );

		float prior_pro( pair<float , float> & x , pair<float, float> & exp , mat2d & sigma );

		void  post_pro();
		/*
 		 * the function calculate the posterior probability of Zi in the condition of Xi
 		 * it's based on Bayes theory, the first component is the probability of the first
 		 * class, so as to the second one. the posterior pro is used to update the mixture of 
 		 * Gaussian model repeatedly
 		 * the E-step, 
 		 */
		void update();
		/*
 		 * update the parameter, the M-step, maximize the likelihood function using the
 		 * post_pro, 
 		 */

	public:
		RedPixelsExtractor(){};
		void initExtractor( vector<float> phai , vector< pair<float , float > > exp , \
												vector< pair<float , float> >features );
		void EMAlgorithm( int iteration_step );
		/*
 		 * to simplify the calculation, we take the max iteration step 
 		 * as iteration step
 		 */
		void takeScore( pair<float , float> & x , vector<float> & score );
		/*
 		 * the function take a feature vector x which contains 2 features, 
 		 * R/B and R/G , 
 		 * it input the computed score in vector score;
 		 * the score is actually the posterior probability
 		 * and it's summed to 1
 		 */
};
#endif
