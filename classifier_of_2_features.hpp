#ifndef CLASSIFIER_OF_2_FEATURES_H_
#define CLASSIFIER_OF_2_FEATURES_H_

#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::pair;
using std::vector;

namespace jh
{
	class classifier
	/*
	 * this basic class is used to classify the red and unred
	 * points, the classifier only take 2 features as input for simplicity
	 * people could use Mixed Gaussian Model or k-Nearest clustering, etc
	 * to do the classification, only by hierachying from the class
	 * and accomplish some functions
	 */
	{
		protected:
			vector< pair<float , float> > features_;
        	/*
         	* features input for learning, the features is filtered by one BINARIZATOR.
         	* users could use other features but the number should be fixed in two.
         	* because higher dimension matrix's inversion should be calculated in BLAS or 
         	* some other software, in this condition, to simplify the calculation we constrain
         	* the calculation in two dimensions
         	*/
		public:
			classifier(){};
			virtual ~classifier(){};
			virtual void takeScore( pair<float , float> & x , vector<float> & score ) = 0;
			/*
         	* the function take a feature vector x containning 2 features, 
         	* which is processed in the same manner as protected component FEATURES_, 
         	* it input the computed score in vector score;
         	* the score is actually the posterior probability
         	* and it's summed to 1
         	*/
			virtual bool isGray( pair<float , float> & x ) = 0;
			/*
        	* this function take a input feature to calculate it defines
         	* a gray point or not, 
         	* first, it calculate the distance of the two expecatation to
         	* point(1,1) as in this class, point(1,1) stands for gray point
			* so, if the expectation is closer to (1,1), this is supposed
        	* to gray class, then we calculate the posterior probability
         	* like before, and selete the probability of which class is
         	* bigger than 0.5;
         	*/
			virtual void info()
			{
				cout << "This is a classifier!\n" << endl;
			}
	};
} // namespace jh

#endif
