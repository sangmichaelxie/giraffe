#ifndef ANN_H
#define ANN_H

#include <array>
#include <algorithm>
#include <random>
#include <functional>
#include <memory>
#include <exception>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <atomic>

#include <cmath>
#include <cassert>

#include "Eigen/Dense"
#include "Eigen/Sparse"

typedef float FP;

typedef Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> NNMatrix;
typedef Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> NNMatrixRM;
typedef Eigen::Matrix<FP, 1, Eigen::Dynamic, Eigen::RowMajor> NNVector;

using Eigen::MatrixBase;

enum ActivationFunc
{
	Linear,
	Tanh,
	Relu
};

inline void ErrorFunc(const NNMatrix &in, NNMatrix &out)
{
	// y = 10 * ln(cosh(x/10))
	for (int32_t i = 0; i < in.rows(); ++i)
	{
		//out(i, 0) = 10.0f * log(cosh(in(i, 0) / 10.0f));
		out(i, 0) = fabs(in(i, 0));
	}
}

inline void ErrorFuncDeri(const NNMatrix &in, NNMatrix &out)
{
	// y = tanh(x/10)
	for (int32_t i = 0; i < in.rows(); ++i)
	{
		//out(i, 0) = tanh(in(i, 0) / 10.0f);
		out(i, 0) = (in(i, 0) > 0.0f) ? 1.0f : -1.0f;
	}
}

template <ActivationFunc ACTF>
class FCANN
{
public:
	// initialize with random weights
	FCANN(
		int randomSeed,
		size_t inputs,
		size_t outputs,
		std::vector<size_t> hiddenLayers,
		std::vector<std::vector<Eigen::Triplet<FP> > > &connectionMatrices);

	struct Activations
	{
		std::vector<NNMatrix> act; // input into each layer
		std::vector<NNMatrix> actIn; // input into activation functions for each layer
	};

	struct Gradients
	{
		std::vector<NNVector> biasGradients;
		std::vector<NNMatrix> weightGradients;

		Gradients &operator+=(const Gradients &other)
		{
			assert(biasGradients.size() == other.biasGradients.size());
			assert(weightGradients.size() == other.weightGradients.size());

			for (size_t i = 0; i < biasGradients.size(); ++i)
			{
				biasGradients[i] += other.biasGradients[i];
				weightGradients[i] += other.weightGradients[i];
			}

			return *this;
		}
	};

	class LearningRateException : public std::runtime_error
	{
	public:
		LearningRateException() : std::runtime_error("Learning rate too high!") {}
	};

	void InitializeActivations(Activations &act);

	void InitializeGradients(Gradients &grad);

	template <typename Derived>
	NNMatrix ForwardPropagate(const MatrixBase<Derived> &in, Activations &act);

	// same as ForwardPropagate, but doesn't bother with Activations
	template <typename Derived>
	NNMatrix ForwardPropagateFast(const MatrixBase<Derived> &in);

	template <typename Derived>
	void BackwardPropagateComputeGrad(const MatrixBase<Derived> &err, const Activations &act, Gradients &grad);

	// this is a convenience function that simply runs 1 iteration of GDM
	template <typename Derived>
	float TrainGDM(const MatrixBase<Derived> &x, const MatrixBase<Derived> &y, float reg);

	void ApplyWeightUpdates(const Gradients &grad, float reg);

	float GetSparsity();

private:

	template <typename Derived>
	void Activate_(MatrixBase<Derived> &x) const;

	template <typename Derived>
	void ActivateDerivative_(MatrixBase<Derived> &x) const;

	void GetThreadBlock_(int64_t numTotal, int64_t &begin, int64_t &num)
	{
		size_t threadId = omp_get_thread_num();
		size_t numThreads = omp_get_num_threads();

		size_t rowsPerThread = numTotal / numThreads;
		size_t rem = numTotal % numThreads; // the first "rem" threads get 1 extra row

		if (threadId < rem)
		{
			begin = threadId * (rowsPerThread + 1);
			num = rowsPerThread + 1;
		}
		else
		{
			begin = rem * (rowsPerThread + 1) + (threadId - rem) * rowsPerThread;
			num = rowsPerThread;
		}
	}

	// this is used to ensure network stability
	constexpr static FP MAX_WEIGHT = 1000.0f;

	// these are network parameters that should be copied by copy ctor and assignment operator
	struct Params
	{
		std::vector<NNVector> outputBias;
		std::vector<NNMatrix> weights;

		std::vector<NNMatrix> weightMasks;

		// the following 2 fields are used by SGD with momentum
		std::vector<NNVector> outputBiasLastUpdate;
		std::vector<NNMatrix> weightsLastUpdate;

		// the following 4 fields are used by ADADELTA
		std::vector<NNVector> outputBiasEg2;
		std::vector<NNMatrix> weightsEg2;
		std::vector<NNVector> outputBiasRMSd2;
		std::vector<NNMatrix> weightsRMSd2;
	} m_params;

	std::mt19937 m_mersenneTwister;
};

#include "ann_impl.h"

#endif // ANN_H
