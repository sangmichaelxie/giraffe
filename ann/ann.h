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

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

typedef float FP;

typedef Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> NNMatrix;
typedef Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> NNMatrixRM;
typedef Eigen::Matrix<FP, 1, Eigen::Dynamic, Eigen::RowMajor> NNVector;

typedef viennacl::matrix<float, viennacl::column_major> VCLMatrix;
typedef viennacl::matrix<float, viennacl::row_major> VCLMatrixRM;
typedef viennacl::vector<float> VCLVector;

enum StorageOrder
{
	RowMajor,
	ColMajor,
	None /* not a matrix */
};

template <typename T>
struct StorageOrderTrait
{
	const static bool value = StorageOrder::None;
};

template<>
struct StorageOrderTrait<NNMatrix>
{
	const static bool value = StorageOrder::ColMajor;
};

template<>
struct StorageOrderTrait<NNMatrixRM>
{
	const static bool value = StorageOrder::RowMajor;
};

template<>
struct StorageOrderTrait<VCLMatrix>
{
	const static bool value = StorageOrder::ColMajor;
};

template<>
struct StorageOrderTrait<VCLMatrixRM>
{
	const static bool value = StorageOrder::RowMajor;
};

using Eigen::MatrixBase;

template <typename EigenType, typename VCLType>
inline void CopyToGPU(const EigenType &cpu, VCLType &gpu)
{
	static_assert(StorageOrderTrait<EigenType>::value != StorageOrder::None, "Not a matrix!");
	static_assert(StorageOrderTrait<VCLType>::value != StorageOrder::None, "Not a matrix!");
	static_assert(StorageOrderTrait<EigenType>::value == StorageOrderTrait<VCLType>::value, "Different storage order!");

	if (static_cast<int64_t>(gpu.size1()) != cpu.rows() || static_cast<int64_t>(gpu.size2()) != cpu.cols())
	{
		gpu.resize(cpu.rows(), cpu.cols());
	}

	// we want to use fast_copy, which means we have to take VCL padding into account
	EigenType padded(gpu.internal_size1(), gpu.internal_size2());
	padded.block(0, 0, cpu.rows(), cpu.cols()) = cpu;

	viennacl::fast_copy(&(padded.data()[0]), &(padded.data()[padded.rows() * padded.cols()]), gpu);
}

template <typename EigenType, typename VCLType>
inline void CopyFromGPU(const VCLType &gpu, EigenType &cpu)
{
	static_assert(StorageOrderTrait<EigenType>::value != StorageOrder::None, "Not a matrix!");
	static_assert(StorageOrderTrait<VCLType>::value != StorageOrder::None, "Not a matrix!");
	static_assert(StorageOrderTrait<EigenType>::value == StorageOrderTrait<VCLType>::value, "Different storage order!");

	EigenType padded(gpu.internal_size1(), gpu.internal_size2());

	viennacl::fast_copy(gpu, &(padded.data()[0]));

	cpu = padded.block(0, 0, gpu.size1(), gpu.size2());
}

// a *= b, using the most efficient algorithm possible (either CPU or GPU)
// we provide gpu temporary matrices to avoid allocation all the time
template <typename EigenTypeA, typename EigenTypeB, typename VCLTypeA, typename VCLTypeB>
inline void Multiply(EigenTypeA &a, const EigenTypeB &b, VCLTypeA &aGpuTmp, VCLTypeB &bGpuTmp, VCLTypeA &resultTmp)
{
	static_assert(StorageOrderTrait<EigenTypeA>::value != StorageOrder::None, "Not a matrix!");
	static_assert(StorageOrderTrait<VCLTypeA>::value != StorageOrder::None, "Not a matrix!");
	static_assert(StorageOrderTrait<EigenTypeA>::value == StorageOrderTrait<VCLTypeA>::value, "Different storage order!");
	static_assert(StorageOrderTrait<EigenTypeB>::value != StorageOrder::None, "Not a matrix!");
	static_assert(StorageOrderTrait<VCLTypeB>::value != StorageOrder::None, "Not a matrix!");
	static_assert(StorageOrderTrait<EigenTypeB>::value == StorageOrderTrait<VCLTypeB>::value, "Different storage order!");

	// matrix multiplication is O(mnp), and requires transferring O(mn + np) amount of data
	// so we only use the GPU is mnp/(mn + np) is above a certain ratio (so that the multiplication is not BW-bound)
	const static int64_t MinComputeRatio = 50;

	int64_t m = a.rows();
	int64_t n = a.cols();
	int64_t p = b.cols();

	int64_t computeRatio = m * n * p / (m * n + n * p);

	if (computeRatio < MinComputeRatio)
	{
		// just do it on the CPU
		a *= b;
	}
	else
	{
		CopyToGPU(a, aGpuTmp);
		CopyToGPU(b, bGpuTmp);

		if (static_cast<int64_t>(resultTmp.size1()) != m || static_cast<int64_t>(resultTmp.size2()) != p)
		{
			resultTmp.resize(m, p);
		}

		resultTmp = viennacl::linalg::prod(aGpuTmp, bGpuTmp);
		CopyFromGPU(resultTmp, a);
	}
}

enum ActivationFunc
{
	Linear,
	Tanh,
	Relu
};

template <typename Derived1, typename Derived2>
inline void ErrorFunc(const MatrixBase<Derived1> &in, MatrixBase<Derived2> &out)
{
	// y = 10 * ln(cosh(x/10))
	for (int32_t i = 0; i < in.rows(); ++i)
	{
		//out(i, 0) = 10.0f * log(cosh(in(i, 0) / 10.0f));
		out(i, 0) = fabs(in(i, 0));
	}
}

template <typename Derived1, typename Derived2>
inline void ErrorFuncDeri(const MatrixBase<Derived1> &in, MatrixBase<Derived2> &out)
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
		std::vector<NNMatrixRM> act; // input into each layer
		std::vector<NNMatrixRM> actIn; // input into activation functions for each layer
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
	NNMatrixRM ForwardPropagate(const MatrixBase<Derived> &in, Activations &act);

	// same as ForwardPropagate, but doesn't bother with Activations
	template <typename Derived>
	NNMatrixRM ForwardPropagateFast(const MatrixBase<Derived> &in);

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

		// these are temporary variables in case we want to do multiplications on GPU
		std::vector<VCLMatrix> weightsGpuTmp;
		std::vector<VCLMatrixRM> weightsTransGpuTmp;
		std::vector<VCLMatrixRM> xGpuTmp;
		std::vector<VCLMatrixRM> errorTermGpuTmp;

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
