#pragma once

#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <set>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>

#include "Eigen/Eigen"
#include "Eigen/SVD"

namespace adrianyu{

	template<class InputIterator, class T>
	T max(InputIterator first, InputIterator last){
		assert(first != last);
		T maxv = *first;
		while (first != last){
			if (maxv < *first){
				maxv = *first;
			}
			++first;
		}
		return maxv;
	}
	template<class InputVec>
	size_t max(const InputVec &alist){
		assert(alist.size() != 0);
		size_t maxidx = 0;
		for (size_t i = 1; i < alist.size(); ++i){
			if (alist[maxidx] < alist[i]){
				maxidx = i;
			}
		}
		return maxidx;
	}
	
	// val must be initialized
	template<class InputIterator, class T>
	void sum(InputIterator first, InputIterator last, T &val){
		while (first != last){
			val += *first;
			++first;
		}
	}

	// compute the Moor-Penrose pseudo-inverse using svd
#ifdef EPSILON
#undef EPSILON
#endif
	// choose the tol wisely
#define EPSILON 1e-6
	template<class MatType>
	int pinv(MatType &inMat, MatType &inMatPinv, typename MatType::Scalar &sProd)
	{
		Eigen::JacobiSVD<MatType> jsvd(inMat, Eigen::ComputeFullU | Eigen::ComputeFullV);
		struct reciprocalNonZero{
			typename MatType::Scalar operator()(typename MatType::Scalar a) const {
				if (std::abs(a) > std::abs(EPSILON)){
					return 1.0 / a;
				}
				else{
					return 0;
				}
			}
		};
		inMatPinv.noalias() = jsvd.matrixV() * jsvd.singularValues().unaryExpr(reciprocalNonZero()).asDiagonal() * jsvd.matrixU().adjoint();

		int nonZeroSingular = 0;
		for (int i = 0; i < jsvd.singularValues().size(); ++i){
			if (std::abs(jsvd.singularValues()(i)) > std::abs(EPSILON)){
				nonZeroSingular++;
			}
		}

		if (nonZeroSingular == 0){
			sProd = 0;
		}
		else{
			struct prodNonZero{
				typename MatType::Scalar operator()(typename MatType::Scalar a) const {
					if (std::abs(a) > std::abs(EPSILON)){
						return a;
					}
					else{
						return 1;
					}
				}
			};
			sProd = jsvd.singularValues().unaryExpr(prodNonZero()).prod();
		}

		return nonZeroSingular;
	}

	// column wise
	template <class MatType>
	int subSample(const MatType &input, MatType &output, const int num){
		if (input.cols() < 1 || num < 1){
			return -1;
		}

		unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
		std::mt19937 rndgen(seed);

		std::vector<int> indices(input.cols());
		for (int i = 0; i < input.cols(); ++i){
			indices[i] = i;
		}
		std::shuffle(indices.begin(), indices.end(), rndgen);

		output.resize(input.rows(), num);
		for (int i = 0; i < num; ++i){
			output.col(i).noalias() = input.col(indices[i]);
		}

		return 0;
	}
	template <class MatType>
	int subSample(const MatType &input, MatType &output, const double ratio){
		//std::uniform_real_distribution<int> rnddist(0.0, 1.0);
		if (input.cols() < 1 || ratio <= 0){
			return -1;
		}
		int outnum = static_cast<int>(static_cast<double>(input.cols()) * ratio);
		if (outnum == 0){
			outnum = 1;
		}

		return subSample(input, output, outnum);
	}


	template<typename Scalar, int DataDim>
	inline Scalar gaussianPDF(const Eigen::Matrix<Scalar, DataDim, 1> &x, const Eigen::Matrix<Scalar, DataDim, 1> &mean, const Eigen::Matrix<Scalar, DataDim, DataDim> &varInv, const Scalar denomiter){
		Scalar pdf = (x - mean).transpose() * varInv * (x - mean);
		pdf = std::exp(-0.5 * pdf) / denomiter;
		//pdf /= std::sqrt(std::pow(2.0*M_PI, x.size()) * varDet);
		return pdf;
	}

	template<typename Scalar, int _DataRows, int _DataCols>
	class GaussianMixture
	{
	public:

		GaussianMixture()
		{
		}

		~GaussianMixture()
		{
		}

		// all data is stored column wise
		int fit(const Eigen::Matrix<Scalar, _DataRows, _DataCols> &data, const int k, std::vector<int> &belongings){
			int sampleNum = data.cols();
			int dataDim = data.rows();
			//std::cout << sampleNum << "\t" << dataDim << std::endl;

			// initialize the size
			belongings.resize(sampleNum);
			weight = Eigen::VectorXd::Zero(k);
			means.resize(dataDim, k);
			variances.resize(k);
			for (int i = 0; i < k; ++i){
				variances[i] = Eigen::Matrix<Scalar, _DataRows, _DataRows>::Zero(dataDim, dataDim);
			}
			varPInv = variances;
			varSingularProd.resize(k);
			pdfDenomiter = varSingularProd;

			std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
			// initialize centers using kmeans++ algorithm
			// then update all parameters.
			int suc = kmeansppInit(data, k, means);
			if (0 != suc){
				return suc;
			}
			std::chrono::duration<double> timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t);
			//std::cout << "done kmeanspp init: " << timeSpan.count() << std::endl;

			//std::cout << means << std::endl;
			#pragma omp parallel for
			for (int i = 0; i < sampleNum; ++i){
				int idx = 0;
				// to speed-up, use l1-norm to replace l2-norm
				// to get accurate results, replace it with:
				// Scalar mindist = (data.col(i) - means.col(0)).squaredNorm();
				Scalar mindist = (data.col(i) - means.col(0)).cwiseAbs().sum();
				for (int j = 1; j < k; ++j){
					// Scalar thisdist = (data.col(i) - means.col(j)).squaredNorm();
					Scalar thisdist = (data.col(i) - means.col(j)).cwiseAbs().sum();
					if (thisdist < mindist){
						mindist = thisdist;
						idx = j;
					}
				}
				belongings[i] = idx;
				#pragma omp critical
				{
					weight(idx)++;
					variances[idx].noalias() += (data.col(i) - means.col(idx))*(data.col(i) - means.col(idx)).transpose();
				}
			}
			weight /= static_cast<Scalar>(sampleNum);
			//std::cout << weight << std::endl;
			for (int i = 0; i < k; ++i){
				variances[i] /= static_cast<Scalar>(sampleNum);
				// using singular products to replace determinant of the variances
				// if the variances is full rank, then the singular products equal to the determinants
				//Scalar sprod;
				int nonZeroSNum = pinv(variances[i], varPInv[i], varSingularProd(i));
				//varSingularProd(i) = sprod;
				pdfDenomiter(i) = std::sqrt(std::pow(2.0*M_PI, nonZeroSNum) * varSingularProd(i));
			}
			timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t);
			//std::cout << "kmeans++ done:" << timeSpan.count() << std::endl;

			// iterate using EM
			Eigen::Matrix<Scalar, _DataCols, Eigen::Dynamic> pprob(sampleNum, k);
			int iter = 0;
			Scalar diff = 0;
			do{
				++iter;
				//std::cout << iter << std::endl;
				t = std::chrono::steady_clock::now();

				// back up the previous param
				Eigen::Matrix<Scalar, _DataRows, Eigen::Dynamic> meanspre = means;
				std::vector< Eigen::Matrix<Scalar, _DataRows, _DataRows> > variancespre = variances;

				// the e step
				getPProbClass(data, pprob, belongings);
				timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t);
				//std::cout << "e-step:" << timeSpan.count() << "seconds" << std::endl;

				// the m step
				weight = pprob.colwise().sum();
				#pragma omp parallel for
				for (int i = 0; i < k; ++i){
					means.col(i).noalias() = (data.cwiseProduct(Eigen::Matrix<Scalar, _DataCols, 1>::Ones(dataDim)*pprob.col(i).transpose())).rowwise().sum() / weight(i);
					variances[i].setZero();
					for (int j = 0; j < sampleNum; ++j){
						variances[i].noalias() += pprob(j, i)*(data.col(j) - means.col(i))*(data.col(j) - means.col(i)).transpose();
					}
					variances[i] /= weight(i);
					int nonZeroSNum = pinv(variances[i], varPInv[i], varSingularProd(i));
					pdfDenomiter(i) = std::sqrt(std::pow(2.0*M_PI, nonZeroSNum) * varSingularProd(i));
					//cout << means.col(i) << endl << variances[i] << endl << varPInv[i] << endl << varSingularProd(i) << endl << endl;
				}
				weight /= sampleNum;
				timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t);
				//std::cout << timeSpan.count() << "seconds" << std::endl;

				// calc the diff, using l1-norm
				diff = (meanspre - means).cwiseAbs().sum() / static_cast<Scalar>(means.rows());
				for (int i = 0; i < k; ++i){
					diff += (variancespre[i] - variances[i]).cwiseAbs().sum() / static_cast<Scalar>(variances[i].size());
				}
				diff /= static_cast<Scalar>(k);

				timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t);
				//std::cout << "m-step: " << timeSpan.count() << "seconds" << std::endl;

			} while (iter < maxiter && diff > tol);

			getPProbClass(data, pprob, belongings);
			Scalar freePramNum = static_cast<Scalar>((dataDim * (dataDim + 1) / 2 + dataDim + 1) * k - 1);
			aic = 2.0 * nlogl + 2.0 * freePramNum;
			aicc = aic + 2.0 * freePramNum * (freePramNum + 1) / (sampleNum - freePramNum - 1);
			bic = 2.0 * nlogl + freePramNum*std::log(static_cast<Scalar>(sampleNum));

			return 0;
		}

		Scalar getAIC(void) const{
			return aic;
		}
		Scalar getAICC(void)const{
			return aicc;
		}
		Scalar getBIC(void)const{
			return bic;
		}
		Scalar getNLogL(void)const{
			return nlogl;
		}

		int getPProbClass(const Eigen::Matrix<Scalar, _DataRows, _DataCols> &data, Eigen::Matrix<Scalar, _DataCols, Eigen::Dynamic> &pprob, std::vector<int> &belongings){
			nlogl = 0;
			int sampleNum = data.cols();
			int k = means.cols();
			pprob.resize(sampleNum, k);
			//Eigen::MatrixXd pprob(sampleNum, k);
			double maxpdf = 1.0 / EPSILON;
			double minpdf = EPSILON;
			#pragma omp parallel for
			for (int i = 0; i < sampleNum; ++i){
				for (int j = 0; j < k; ++j){
					if (varSingularProd(j) > EPSILON){
						pprob(i, j) = gaussianPDF<Scalar, _DataRows>(data.col(i), means.col(j), varPInv[j], pdfDenomiter(j));
					}
					else{
						// if clustering data is (almost) distinct
						double dist = (data.col(i) - means.col(j)).cwiseAbs().sum();
						if (dist < EPSILON){
							pprob(i, j) = maxpdf;
						}
						else{
							pprob(i, j) = minpdf;
						}
					}
					pprob(i, j) *= weight(j);
				}
				Scalar rsum = pprob.row(i).sum();
				nlogl -= std::log(rsum);
				pprob.row(i) /= rsum;
				belongings[i] = max(pprob.row(i));
			}

			return 0;
		}

	private:
		Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weight; // mixture weight
		Eigen::Matrix<Scalar, _DataRows, Eigen::Dynamic> means;	// 
		std::vector< Eigen::Matrix<Scalar, _DataRows, _DataRows> > variances;
		std::vector< Eigen::Matrix<Scalar, _DataRows, _DataRows> > varPInv;
		Eigen::Matrix<Scalar, Eigen::Dynamic, 1> varSingularProd;
		Eigen::Matrix<Scalar, Eigen::Dynamic, 1> pdfDenomiter;
		const int maxiter = 100;
		const Scalar tol = 1e-4;
		Scalar nlogl;
		Scalar aic;
		Scalar aicc;
		Scalar bic;


		int kmeansppInit(const Eigen::Matrix<Scalar, _DataRows, _DataCols> &data, const int k, Eigen::Matrix<Scalar, _DataRows, Eigen::Dynamic> &centers){
			const int samplenNum = data.cols();
			const int dataDim = data.rows();

			centers.resize(dataDim, k);
			//std::vector<int> indices;

			std::uniform_real_distribution<Scalar> rnddist(0.0, 1.0);
			unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
			std::mt19937 rndgen(seed);

			int idx = static_cast<int>(std::floor(static_cast<Scalar>(samplenNum)*rnddist(rndgen)));
			if (idx < 0){
				idx = 0;
			}
			if (idx >= samplenNum){
				idx = samplenNum - 1;
			}

			// assign the first center
			centers.col(0).noalias() = data.col(idx);
			//indices.push_back(idx);
			std::vector<Scalar> distances(samplenNum, 0.0);
			std::vector<Scalar> maxdists(samplenNum, 0.0);
			for (int i = 1; i < k; ++i){
				// querying closest center and get the distance
				#pragma omp parallel for
				for (int j = 0; j < samplenNum; ++j){
					// using l1-norm to replace l2-norm
					// Scalar mindist = (data.col(j) - centers.col(0)).squaredNorm(); // the original operation
					Scalar mindist = (data.col(j) - centers.col(0)).cwiseAbs().sum();
					Scalar maxdist = mindist;
					for (int m = 1; m < i; ++m){
						//Scalar adisttmp = (data.col(j) - centers.col(m)).squaredNorm();
						Scalar adisttmp = (data.col(j) - centers.col(m)).cwiseAbs().sum();
						if (adisttmp < mindist){
							mindist = adisttmp;
						}
						if (adisttmp > maxdist){
							maxdist = adisttmp;
						}
					}
					distances[j] = mindist;
					maxdists[j] = maxdist;
				}
				//std::cout << i << std::endl;
				// just in case that there are less distinct data samples than centers.
				Scalar maxd = max<typename std::vector<Scalar>::iterator, Scalar>(maxdists.begin(), maxdists.end());
				if (maxd < EPSILON){
					return -1;
				}

				// get next center
				// choose the point using a weight probability distribution
				// where a point x is chosen with probability proportional to squared distance.
				Scalar distsum = 0.0;
				sum(distances.begin(), distances.end(), distsum);
				do{
					Scalar posf = distsum * rnddist(rndgen);
					Scalar cpos = 0;
					idx = samplenNum - 1;
					for (size_t j = 0; j < distances.size(); ++j){
						cpos += distances[j];
						if (cpos > posf){
							idx = j;
							break;
						}
					}
				} while (distances[idx] < EPSILON);
				//indices.push_back(idx);
				centers.col(i).noalias() = data.col(idx);
			}
			return 0;
		}
	};

}
