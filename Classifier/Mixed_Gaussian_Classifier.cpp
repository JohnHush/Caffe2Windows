#include "Mixed_Gaussian_Classifier.hpp"
#include <iostream>
#include <stdlib.h>
#include <cmath>

using std::cout;
using std::endl;

namespace jh
{
	void mg_classifier::initExtractor( vector< pair<float , float> >features )
	{                                                                                                                      
		int data_num = features.size();

		features_.resize( data_num );
		phai_.resize(2);
		sigma_.resize(2);
		exp_.resize(2);
		post_.resize( data_num );

		phai_[0] = 0.5;
		phai_[1] = 0.5;

		/*
		 * compute the std for the whole data
		 * and set every sigma to this std as an inital value
		 */
		vector<float> ave(2, 0.);


		for ( int i = 0 ; i < data_num ; ++i )
		{
			ave[0] += features[i].first;
			ave[1] += features[i].second;
		}
		ave[0] /= data_num;
		ave[1] /= data_num;

		vector<float> std_( 2 , 0.);

		for ( int i = 0 ; i < data_num ; ++i )
		{
			std_[0] += ( ave[0] - features[i].first )  * ( ave[0] - features[i].first );
			std_[1] += ( ave[1] - features[i].second ) * ( ave[1] - features[i].second );
		}
		std_[0] /= data_num;
		std_[1] /= data_num;

		sigma_[0].a00 = std_[0];
		sigma_[0].a11 = std_[1];
		sigma_[0].a01 = 0.;
		sigma_[0].a10 = 0.;

		sigma_[1].a00 = std_[0];
		sigma_[1].a11 = std_[1];
		sigma_[1].a01 = 0.;
		sigma_[1].a10 = 0.;

		for ( int i = 0 ; i < features.size() ; ++ i)
		{
			features_[i].first  = features[i].first;
			features_[i].second = features[i].second;
		}
		std_[0] = sqrt( std_[0] );
		std_[1] = sqrt( std_[1] );

		float set_range = 2.8 ;
		float pt_dis = 0.1;
		// the point is set in ave+_set_range*std;
		bool OUTOFRANGE;
		bool TOOCLOSE;
		do
		{
			int index1 = int((double(rand())/RAND_MAX) * (data_num-1) );
			int index2 = int((double(rand())/RAND_MAX) * (data_num-1) );

			exp_[0] = features_[index1];
			exp_[1] = features_[index2];

			OUTOFRANGE = false;
			TOOCLOSE   = false;

			if ( exp_[0].first < ave[0] - set_range * std_[0] || exp_[0].first > ave[0] + set_range * std_[0] || \
					exp_[1].first < ave[0] - set_range * std_[0] || exp_[1].first > ave[0] + set_range * std_[0] || \
					exp_[0].second< ave[1] - set_range * std_[1] || exp_[0].second> ave[1] + set_range * std_[1] || \
					exp_[1].second< ave[1] - set_range * std_[1] || exp_[1].second> ave[1] + set_range * std_[1] )
				OUTOFRANGE = true;
			if ( fabs( exp_[0].first - exp_[1].first ) < pt_dis * std_[0] || \
					fabs( exp_[0].second- exp_[1].second) < pt_dis * std_[1])
				TOOCLOSE = true;
		}
		while( OUTOFRANGE || TOOCLOSE );
	}

	void mg_classifier::initExtractor( vector<float> phai , vector< pair<float , float> > miu ,\
			vector< pair<float , float> >features )
	{
		int data_num = features.size();

		features_.resize( data_num );
		phai_.resize(2);
		sigma_.resize(2);
		exp_.resize(2);
		post_.resize( data_num );

		phai_[0] = phai[0];
		phai_[1] = phai[1];

		exp_[0]  = miu[0];
		exp_[1]  = miu[1];

		/*
		 * compute the std for the whole data
		 * and set every sigma to this std as an inital value
		 */
		vector<float> ave(2, 0.);

		for ( int i = 0 ; i < data_num ; ++i )
		{
			ave[0] += features[i].first;
			ave[1] += features[i].second;
		}
		ave[0] /= data_num;
		ave[1] /= data_num;

		vector<float> std_( 2 , 0.);

		for ( int i = 0 ; i < data_num ; ++i )
		{
			std_[0] += ( ave[0] - features[i].first )  * ( ave[0] - features[i].first );
			std_[1] += ( ave[1] - features[i].second ) * ( ave[1] - features[i].second );
		}
		std_[0] /= data_num;
		std_[1] /= data_num;

		sigma_[0].a00 = std_[0];
		sigma_[0].a11 = std_[1];
		sigma_[0].a01 = 0.;
		sigma_[0].a10 = 0.;

		sigma_[1].a00 = std_[0];
		sigma_[1].a11 = std_[1];
		sigma_[1].a01 = 0.;
		sigma_[1].a10 = 0.;

		for ( int i = 0 ; i < features.size() ; ++ i)
		{
			features_[i].first  = features[i].first;
			features_[i].second = features[i].second;                                                                      
		}
	}

	void mg_classifier::post_pro()
	{
		int data_num = features_.size();

		for ( int iDATA = 0 ; iDATA < data_num ; ++ iDATA )
		{
			float first_prior = prior_pro_2d( features_[iDATA] , exp_[0] , sigma_[0] );
			float secon_prior = prior_pro_2d( features_[iDATA] , exp_[1] , sigma_[1] );

			if ( first_prior < 1E-15 && secon_prior < 1E-15 )
			{
				post_[iDATA].first  = 0.5 + (-50 + rand()%100)/200. ;
				post_[iDATA].second = 1. - post_[iDATA].first;
			}
			/*
			 * if the two prior probabilitys are too small, then we
			 * set all the posterior probability around 0.5 to avoid
			 * nan value
			 */
			else
			{
				post_[iDATA].first   = ( first_prior * phai_[0] ) / ( first_prior * phai_[0] + secon_prior * phai_[1] );
				post_[iDATA].second  = ( secon_prior * phai_[1] ) / ( first_prior * phai_[0] + secon_prior * phai_[1] );
			}
		}
	}

	void mg_classifier::update()
	{
		vector<float> phai_before(2);
		vector< pair<float , float> > exp_before(2);

		phai_before[0] = phai_[0];
		phai_before[1] = phai_[1];

		exp_before[0] = exp_[0];
		exp_before[1] = exp_[1];

		int data_num = features_.size();

		phai_[0] = 0.;
		phai_[1] = 0.;

		for ( int iDATA = 0 ; iDATA < data_num ; ++ iDATA )
		{
			phai_[0] += post_[iDATA].first;
			phai_[1] += post_[iDATA].second;
		}
		phai_[0] /= data_num;
		phai_[1] /= data_num;

		/*
		 * phai_j = (1/m) * sum over i to m ( post_i_j )
		 */
		exp_[0].first  = 0.;
		exp_[0].second = 0.;
		exp_[1].first  = 0.;
		exp_[1].second = 0.;

		for ( int iDATA = 0 ; iDATA < data_num ; ++ iDATA )
		{
			exp_[0].first  += post_[iDATA].first  * features_[iDATA].first;
			exp_[0].second += post_[iDATA].first  * features_[iDATA].second;
			exp_[1].first  += post_[iDATA].second * features_[iDATA].first;
			exp_[1].second += post_[iDATA].second * features_[iDATA].second;
		}
		exp_[0].first  /= ( data_num * phai_[0] );
		exp_[0].second /= ( data_num * phai_[0] );
		exp_[1].first  /= ( data_num * phai_[1] );
		exp_[1].second /= ( data_num * phai_[1] );
		/*
		 * exp_j = ( sum over i( post_i_j * features_i ) ) / ( sum over i( post_i_j ) );
		 */

		sigma_[0].a00 = 0.; sigma_[0].a01 = 0.; sigma_[0].a10 = 0.; sigma_[0].a11 = 0;
		sigma_[1].a00 = 0.; sigma_[1].a01 = 0.; sigma_[1].a10 = 0.; sigma_[1].a11 = 0;

		pair<float , float> tmp1;
		pair<float , float> tmp2;
		// tmp = x_i - u_j;
		for ( int iDATA = 0 ; iDATA < data_num ; ++ iDATA )
		{
			tmp1.first  = features_[iDATA].first  - exp_[0].first;
			tmp1.second = features_[iDATA].second - exp_[0].second;
			tmp2.first  = features_[iDATA].first  - exp_[1].first;
			tmp2.second = features_[iDATA].second - exp_[1].second;

			sigma_[0].a00 += post_[iDATA].first * tmp1.first  * tmp1.first;
			sigma_[0].a01 += post_[iDATA].first * tmp1.first  * tmp1.second;
			sigma_[0].a10 += post_[iDATA].first * tmp1.first  * tmp1.second;
			sigma_[0].a11 += post_[iDATA].first * tmp1.second * tmp1.second;

			sigma_[1].a00 += post_[iDATA].second * tmp2.first  * tmp2.first;
			sigma_[1].a01 += post_[iDATA].second * tmp2.first  * tmp2.second;
			sigma_[1].a10 += post_[iDATA].second * tmp2.first  * tmp2.second;
			sigma_[1].a11 += post_[iDATA].second * tmp2.second * tmp2.second;
		}
		sigma_[0].a00 /= ( data_num * phai_[0] );
		sigma_[0].a01 /= ( data_num * phai_[0] );
		sigma_[0].a10 /= ( data_num * phai_[0] );
		sigma_[0].a11 /= ( data_num * phai_[0] );

		sigma_[1].a00 /= ( data_num * phai_[1] );
		sigma_[1].a01 /= ( data_num * phai_[1] );
		sigma_[1].a10 /= ( data_num * phai_[1] );
		sigma_[1].a11 /= ( data_num * phai_[1] );

		delta_phai_.resize(2);
		delta_exp_.resize(2);

		delta_phai_[0] = phai_[0] - phai_before[0];
		delta_phai_[1] = phai_[1] - phai_before[1];

		delta_exp_[0].first  = exp_[0].first  - exp_before[0].first;
		delta_exp_[0].second = exp_[0].second - exp_before[0].second;
		delta_exp_[1].first  = exp_[1].first  - exp_before[1].first;
		delta_exp_[1].second = exp_[1].second - exp_before[1].second;

	}

	void mg_classifier::EMAlgorithm( const int iteration_step )
	{
		float epsilon = 1E-10;
		float CRatio  = 0.00001;
		for ( int i = 0 ; i < iteration_step ; ++ i )
		{
			post_pro();
			update();

			if (    fabs( delta_phai_[0] )      / ( fabs(phai_[0]) + epsilon )      < CRatio  && 
					fabs( delta_phai_[1] )      / ( fabs(phai_[1]) + epsilon )      < CRatio && 
					fabs( delta_exp_[0].first ) / ( fabs(exp_[0].first) + epsilon ) < CRatio && 
					fabs( delta_exp_[0].second )/ ( fabs(exp_[0].second) + epsilon )< CRatio &&
					fabs( delta_exp_[1].first ) / ( fabs(exp_[1].first) + epsilon ) < CRatio &&
					fabs( delta_exp_[1].second )/ ( fabs(exp_[1].second) + epsilon )< CRatio )
				break;
		}
	}

	void mg_classifier::info()
	{
		cout << "This is a Mixed Gaussian Classifier" << endl;
		cout << "prior probability of first class is = " << phai_[0] << endl;
		cout << "prior probability of secon class is = " << phai_[0] << endl;
		cout << "SIGMA INFORMATION " << endl << "first class: " << endl;
		cout << "\t\t a00 = " << sigma_[0].a00 << endl;
		cout << "\t\t a01 = " << sigma_[0].a01 << endl;
		cout << "\t\t a10 = " << sigma_[0].a10 << endl;
		cout << "\t\t a11 = " << sigma_[0].a11 << endl;
		cout << "second class: " << endl;
		cout << "\t\t a00 = " << sigma_[1].a00 << endl;
		cout << "\t\t a01 = " << sigma_[1].a01 << endl;
		cout << "\t\t a10 = " << sigma_[1].a10 << endl;
		cout << "\t\t a11 = " << sigma_[1].a11 << endl;
		cout << "expectation of first class is = [ " << exp_[0].first << "  ,  " << exp_[0].second << " ]" << endl;
		cout << "expectation of secon class is = [ " << exp_[1].first << "  ,  " << exp_[1].second << " ]" << endl;
	}

	void mg_classifier::takeScore( pair<float , float> &x , vector<float> & score )
	{ 
		float first_prior = prior_pro_2d( x , exp_[0] , sigma_[0] );
		float secon_prior = prior_pro_2d( x , exp_[1] , sigma_[1] );

		score.resize(2);

		score[0] = ( first_prior * phai_[0] ) / ( first_prior * phai_[0] + secon_prior * phai_[1] );
		score[1] = 1. - score[0];
	}

	bool mg_classifier::isGray( pair<float , float> & x )
	{
		float dis0 = (exp_[0].first - 1. ) * (exp_[0].second - 1. );
		float dis1 = (exp_[1].first - 1. ) * (exp_[1].second - 1. );

		int gray_index = dis0>dis1?1:0;
		int colo_index = 1 - gray_index;
		// decide which label is gray
		//
		vector<float> prior(2);

		prior[0] = prior_pro_2d( x , exp_[0] , sigma_[0] );
		prior[1] = prior_pro_2d( x , exp_[1] , sigma_[1] );

		if ( ( prior[gray_index] * phai_[gray_index] ) / \
				( prior[gray_index] * phai_[gray_index] + prior[colo_index] * phai_[colo_index]  ) > 0.5 )
			return true;

		return false;
	}

	float mg_classifier::relativeLikelihood( const float MP )
	{
		float LIKELIHOOD = 0.;

		for ( int iDATA = 0 ; iDATA < features_.size() ; ++ iDATA )
		{
			float prior0 = prior_pro_2d( features_[iDATA] , exp_[0] , sigma_[0] );
			float prior1 = prior_pro_2d( features_[iDATA] , exp_[1] , sigma_[1] );

			LIKELIHOOD += log( MP * ( prior0 * phai_[0] + prior1 * phai_[1] ) );

		}
		return LIKELIHOOD;
	}

} // namespace jh
