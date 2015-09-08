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
#include <ostream>
#include <istream>

#include <cmath>
#include <cassert>

#include "matrix_ops.h"

enum ActivationFunc
{
	Linear,
	Tanh,
	Relu,
	Softmax,
	Logsig
};

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
class FCANN
{
public:
	FCANN() {}

	// initialize with random weights
	FCANN(
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

	// same as ForwardPropagate, but doesn't bother with Activations (NOT REENTRANT!!)
	template <typename Derived>
	NNMatrixRM ForwardPropagateFast(const MatrixBase<Derived> &in);

	// special case for 1 board and single-valued output - this is used in gameplay (NOT REENTRANT!!)
	template <typename Derived>
	float ForwardPropagateSingle(const MatrixBase<Derived> &vec);

	// special case for eval while also reading out signature
	template <typename Derived>
	float ForwardPropagateSingleWithSignature(const MatrixBase<Derived> &vec, float *signOut);

	template <typename Derived>
	void BackwardPropagateComputeGrad(const MatrixBase<Derived> &err, const Activations &act, Gradients &grad);

	// this is a convenience function that simply runs 1 iteration of GDM
	template <typename Derived1, typename Derived2>
	float TrainGDM(const MatrixBase<Derived1> &x, const MatrixBase<Derived2> &y, float learningRate, float reg);

	void ApplyWeightUpdates(const Gradients &grad, float learningRate, float reg);

	float GetSparsity();

	typedef NNVector BiasType;
	typedef NNMatrix WeightType;
	typedef NNMatrix WeightMaskType;

	// these are used to save and restore nets
	std::vector<BiasType> &Biases() { return m_params.outputBias; }
	std::vector<WeightType> &Weights() { m_params.weightsSemiSparseCurrent = false; return m_params.weights; }
	std::vector<WeightMaskType> &WeightMasks() { return m_params.weightMasks; }

	void NotifyWeightMasksChanged() { UpdateWeightMasksRegions_(); }

	int64_t OutputCols() const { return m_params.weights[m_params.weights.size() - 1].cols(); }

	template <typename Derived1, typename Derived2>
	NNMatrixRM ErrorFunc(const MatrixBase<Derived1> &pred, const MatrixBase<Derived2> &targets) const;

	template <typename Derived1, typename Derived2, typename Derived3>
	NNMatrixRM ErrorFuncDerivative(const MatrixBase<Derived1> &pred, const MatrixBase<Derived2> &targets, const MatrixBase<Derived3> &finalLayerActivations) const;

private:
	template <typename Derived>
	void Activate_(MatrixBase<Derived> &x, bool last) const;

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

	void UpdateWeightMasksRegions_();

	void UpdateWeightSemiSparse_();

	// this is used to ensure network stability
	constexpr static FP MAX_WEIGHT = 1000.0f;

	// these are network parameters that should be copied by copy ctor and assignment operator
	struct Params
	{
		// bias, weights, and weightMasks completely define the net
		std::vector<BiasType> outputBias;
		std::vector<WeightType> weights;
		std::vector<WeightMaskType> weightMasks;

		// optimized form of weight masks (in lists of regions)
		std::vector<std::vector<MatrixRegion> > weightMasksRegions;

		// optimized form of weight matrices (semi-sparse)
		bool weightsSemiSparseCurrent;
		std::vector<SemiSparseMatrix<WeightType>> weightsSemiSparse;

		// these are temporary variables for evaluating the net, so we don't have to keep allocating and de-allocating
		std::vector<NNMatrixRM> evalTmp;
		std::vector<NNVector> evalSingleTmp;

		// the following 2 fields are used by SGD with momentum
		std::vector<NNVector> outputBiasLastUpdate;
		std::vector<NNMatrix> weightsLastUpdate;

		// the following 4 fields are used by ADADELTA
		std::vector<NNVector> outputBiasEg2;
		std::vector<NNMatrix> weightsEg2;
		std::vector<NNVector> outputBiasRMSd2;
		std::vector<NNMatrix> weightsRMSd2;
	} m_params;
};

typedef FCANN<Relu, Tanh> EvalNet;
typedef FCANN<Relu, Logsig> MoveEvalNet;

template <typename T>
void SerializeNet(T &net, std::ostream &s);

template <typename T>
void DeserializeNet(T &net, std::istream &s);

#include "ann_impl.h"

#endif // ANN_H
