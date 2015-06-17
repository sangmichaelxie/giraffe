#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

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
inline void MultiplyGPU(EigenTypeA &a, const EigenTypeB &b, VCLTypeA &aGpuTmp, VCLTypeB &bGpuTmp, VCLTypeA &resultTmp)
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

#endif // MATRIX_OPS_H
