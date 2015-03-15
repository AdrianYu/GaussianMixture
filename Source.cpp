#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <utility>
#include <vector>
#include <mutex>
#include <regex>
#include <memory>
#include <thread>
#include <string>

#include "Eigen/Eigen"
#include "Eigen/SVD"

#include "GaussianMixture.h"

using namespace std;
using namespace Eigen;
using namespace adrianyu;

#define ROWS 3

int main(void)
{
	int colnum = 20000;
	GaussianMixture<double, ROWS, Dynamic> gmm;

	Matrix<double, ROWS, Dynamic> points(ROWS, colnum);
	ifstream inputf("./data");
	for (int i = 0; i < colnum; ++i){
		for (int j = 0; j < ROWS; ++j){
			inputf >> points(j, i);
		}
	}
	vector<int> belongings;
	vector<double> aics(6);
	int k = 6;
	for (int i = 1; i < 7; ++i){
		cout << i << "th fit...................." << endl;
		gmm.fit(points, i, belongings);
		double aaic = gmm.getAIC();
		aics[i - 1] = aaic;
		cout << "aic:" << aics[i - 1] << endl;
		gmm.fit(points, i, belongings);
		if (gmm.getAIC() < aaic){
			aics[i - 1] = gmm.getAIC();
		}
		cout << "aic:" << aics[i - 1] << endl;
		if (i > 1 && aics[i-1]>aics[i-2]){
			k = i - 1;
			break;
		}
	}

	vector<int> bclass;
	gmm.fit(points, k, belongings);
	bclass = belongings;
	double aaic = gmm.getAIC();
	gmm.fit(points, k, belongings);
	if (gmm.getAIC() < aaic){
		bclass = belongings;
	}

	ofstream outf("./classes");
	for (size_t i = 0; i < bclass.size(); ++i){
		outf << bclass[i] << endl;
	}

	return 0;
}




