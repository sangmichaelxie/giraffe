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

	// we have to define our own copy and assignment ctors because threads and thread controls need
	// to be reconstructed
	FCANN &operator=(const FCANN &other);
	FCANN(const FCANN &other) { *this = other; }

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

	// compute threads do a forward + backward pass on a set of examples, and return
	// the gradients
	struct ComputeThreadControl
	{
		ComputeThreadControl() :
			exit(false), run(false) {} // everything else can be default-constructed

		const NNMatrix *x;
		const NNMatrix *y;

		int64_t rowBegin;
		int64_t numRows;

		Gradients grad;
		NNMatrix errorsMeasure;
		bool exit;
		bool run;

		std::mutex mtx;
		std::condition_variable cv;
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
	float TrainGDM(const NNMatrix &x, const NNMatrix &y, float reg);

	void ApplyWeightUpdates(const Gradients &grad, float reg);

	void SetNumThreads(size_t n);

	float GetSparsity();

	~FCANN() { SetNumThreads(0); } // this joins and deletes all compute threads

private:

	template <typename Derived>
	void Activate_(MatrixBase<Derived> &x) const;

	template <typename Derived>
	void ActivateDerivative_(MatrixBase<Derived> &x) const;

	void ComputeThreadMain_(ComputeThreadControl *ctrl);

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

	// we have to use unique_ptr here because thread synchronization primitives
	// cannot be std::move()'ed
	std::vector<std::unique_ptr<ComputeThreadControl> > m_threadControls;
	std::vector<std::thread> m_computeThreads;
};

#include "ann_impl.h"

#endif // ANN_H
