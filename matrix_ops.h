/*
	Copyright (C) 2015 Matthew Lai

	Giraffe is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
	
	Giraffe is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	
	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <vector>

#include "Eigen/Dense"
#include "Eigen/Sparse"

typedef float FP;

typedef Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> NNMatrix;
typedef Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> NNMatrixRM;
typedef Eigen::Matrix<FP, 1, Eigen::Dynamic, Eigen::RowMajor> NNVector;

using Eigen::MatrixBase;

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
	c = EigenC::Zero(a.rows(), b.cols);

	assert(a.rows() == 1);

	for (const auto &subMatrix : b.subMatrices)
	{
		c.segment(subMatrix.j, subMatrix.m.cols()) += a.segment(subMatrix.i, subMatrix.m.rows()) * subMatrix.m;
	}
}

template <typename EigenA, typename EigenB, typename EigenC>
void MatrixMultiplyWithSemiSparse(const EigenA &a, const SemiSparseMatrix<EigenB> &b, EigenC &c)
{
	// c = a * b
	c = EigenC::Zero(a.rows(), b.cols);

	for (const auto &subMatrix : b.subMatrices)
	{
		c.block(0, subMatrix.j, c.rows(), subMatrix.m.cols()) += a.block(0, subMatrix.i, a.rows(), subMatrix.m.rows()) * subMatrix.m;
	}
}

#endif // MATRIX_OPS_H
