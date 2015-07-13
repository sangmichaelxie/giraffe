#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <vector>

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

struct MatrixRegion
{
	int64_t i;
	int64_t j;
	int64_t rows;
	int64_t cols;
};

template <typename T>
struct SemiSparseMatrix
{
	int64_t rows;
	int64_t cols;

	struct SubMatrix
	{
		int64_t i;
		int64_t j;
		T m;
	};

	std::vector<SubMatrix> subMatrices;
};

template <typename T>
std::vector<MatrixRegion> MatrixToRegions(T toConvert) // matrix passed by value since we need a copy to modify anyways
{
	std::vector<MatrixRegion> ret;

	while (true)
	{
		MatrixRegion newRegion;
		bool nonZeroFound = false;

		// find the first nonzero
		for (int64_t i = 0; i < toConvert.rows(); ++i)
		{
			for (int64_t j = 0; j < toConvert.cols(); ++j)
			{
				// we are looking for exact zeros, so we don't need to check with threshold
				if (toConvert(i, j) != 0.0f)
				{
					newRegion.i = i;
					newRegion.j = j;
					nonZeroFound = true;
					break;
				}
			}

			if (nonZeroFound)
			{
				break;
			}
		}

		if (!nonZeroFound)
		{
			// the matrix is all zero, so we are done!
			break;
		}

		newRegion.rows = 0;
		newRegion.cols = 0;

		// try to grow in rows (only need to check 1 element at a time, since we are growing from a single element)
		while ((newRegion.i + newRegion.rows) < toConvert.rows() && toConvert(newRegion.i + newRegion.rows, newRegion.j) != 0.0f)
		{
			++newRegion.rows;
		}

		// try to grow in cols (need to check 1 vector at a time)
		while ((newRegion.j + newRegion.cols) < toConvert.cols() && toConvert.block(newRegion.i, newRegion.j + newRegion.cols, newRegion.rows, 1).all())
		{
			++newRegion.cols;
		}

		ret.push_back(newRegion);
		assert(toConvert.block(newRegion.i, newRegion.j, newRegion.rows, newRegion.cols).all());
		toConvert.block(newRegion.i, newRegion.j, newRegion.rows, newRegion.cols).setZero();
	}

	return ret;
}

template <typename T>
SemiSparseMatrix<T> ToSemiSparse(const T &m, const std::vector<MatrixRegion> &rois)
{
	SemiSparseMatrix<T> ret;

	ret.rows = m.rows();
	ret.cols = m.cols();

	for (const auto &roi : rois)
	{
		typename
		SemiSparseMatrix<T>::SubMatrix subm;

		subm.i = roi.i;
		subm.j = roi.j;
		subm.m = m.block(roi.i, roi.j, roi.rows, roi.cols);

		ret.subMatrices.push_back(subm);
	}

	return ret;
}

template <typename EigenA, typename EigenB, typename EigenC>
void MultiplyWithSemiSparse(const EigenA &a, const SemiSparseMatrix<EigenB> &b, EigenC &c)
{
	// c = a * b

	// special case for single region
	if (b.subMatrices.size() == 1)
	{
		c.noalias() = a * b.subMatrices[0].m;
		return;
	}

	c = EigenC::Zero(a.rows(), b.cols);

	assert(a.rows() == 1);

	for (const auto &subMatrix : b.subMatrices)
	{
		for (int64_t col = 0; col < subMatrix.m.cols(); ++col)
		{
			c(0, col + subMatrix.j) = a.block(0, subMatrix.i, 1, subMatrix.m.rows()).dot(subMatrix.m.col(col).transpose());
		}
	}
}

#endif // MATRIX_OPS_H
