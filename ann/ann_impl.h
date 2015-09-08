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

#include "ann.h"

#include <iostream>
#include <memory>
#include <algorithm>

#include <cstdint>

#include <omp.h>

// for floating point interrupts
#include <xmmintrin.h>

#include "omp_scoped_thread_limiter.h"
#include "random_device.h"

inline void EnableNanInterrupt()
{
	_MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
FCANN<ACTF, ACTFLast>::FCANN(
	size_t inputs,
	size_t outputs,
	std::vector<size_t> hiddenLayers,
	std::vector<std::vector<Eigen::Triplet<FP> > > &connectionMatrices)
{
	if (connectionMatrices.size() != (hiddenLayers.size() + 1))
	{
		throw std::runtime_error("connectionMatrices.size() should be hiddenLayers.size() + 1");
	}

	auto mt = gRd.MakeMT();

	// then we build the weight and bias vectors, and initialize them
	for (size_t layer = 0; layer < (hiddenLayers.size() + 1); ++layer)
	{
		size_t in_size = (layer == 0) ? inputs : hiddenLayers[layer - 1];
		size_t out_size = (layer == hiddenLayers.size()) ? outputs : hiddenLayers[layer];

		NNMatrix weightMatrix(in_size, out_size);
		NNVector biasVector(out_size);

		// determine what distribution to use to initialize hidden weights depending on activation func
		// (output layer is always linear)
		std::function<FP()> drawFunc;

		if (ACTF == Linear || layer == hiddenLayers.size())
		{
			std::uniform_real_distribution<FP> dist(-0.01f, 0.01f);
			drawFunc = std::bind(dist, mt);
		}
		else if (ACTF == Tanh)
		{
			// for tanh, we use r = sqrt(6/(fan_in + fan_out)), (-r, r)
			FP r = sqrt(6.0/(in_size + out_size));
			std::uniform_real_distribution<FP> dist(-r, r);
			drawFunc = std::bind(dist, mt);
		}
		else if (ACTF == Relu)
		{
			// we use the scheme described here - http://arxiv.org/pdf/1502.01852v1.pdf
			std::normal_distribution<FP> dist(0.0f, sqrt(2.0f/out_size));
			drawFunc = std::bind(dist, mt);
		}
		else
		{
			assert(false);
		}

		// actually initialize weights and biases
		for (size_t j = 0; j < out_size; ++j)
		{
			biasVector(j) = 0.0f;
		}

		for (size_t i = 0; i < in_size; ++i)
		{
			for (size_t j = 0; j < out_size; ++j)
			{
				weightMatrix(i, j) = drawFunc();
			}
		}

		m_params.outputBias.push_back(biasVector);
		m_params.weights.push_back(weightMatrix);

		if (connectionMatrices[layer].size() != 0)
		{
			// we have a sparse layer
			NNMatrix conn = NNMatrix::Zero(in_size, out_size);

			for (const auto &trip : connectionMatrices[layer])
			{
				conn(trip.row(), trip.col()) = 1.0f;
			}

			m_params.weightMasks.push_back(conn);
		}
		else
		{
			// we have a fully connected layer
			m_params.weightMasks.push_back(NNMatrix::Ones(in_size, out_size));
		}

		m_params.outputBiasLastUpdate.push_back(NNVector::Zero(out_size));
		m_params.weightsLastUpdate.push_back(NNMatrix::Zero(in_size, out_size));

		m_params.outputBiasEg2.push_back(NNVector::Zero(out_size));
		m_params.weightsEg2.push_back(NNMatrix::Zero(in_size, out_size));

		m_params.outputBiasRMSd2.push_back(NNVector::Zero(out_size));
		m_params.weightsRMSd2.push_back(NNMatrix::Zero(in_size, out_size));
	}

	m_params.evalTmp.resize(hiddenLayers.size() + 2);
	m_params.evalSingleTmp.resize(hiddenLayers.size() + 2);

	UpdateWeightMasksRegions_();
	UpdateWeightSemiSparse_();
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
void FCANN<ACTF, ACTFLast>::InitializeActivations(Activations &act)
{
	assert(m_params.weights.size() == m_params.outputBias.size());

	act.act.clear();

	act.actIn.clear();

	// we don't know how many rows these matrices will have yet, since
	// that depends on input batch size
	for (size_t layer = 0; layer < m_params.weights.size(); ++layer)
	{
		act.act.push_back(NNVector::Zero(1, m_params.weights[layer].rows()));
		act.actIn.push_back(NNVector::Zero(1, m_params.weights[layer].rows()));
	}

	act.act.push_back(NNVector::Zero(1, m_params.weights[m_params.weights.size() - 1].cols()));
	act.actIn.push_back(NNVector::Zero(1, m_params.weights[m_params.weights.size() - 1].cols()));
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
void FCANN<ACTF, ACTFLast>::InitializeGradients(Gradients &grad)
{
	assert(m_params.weights.size() == m_params.outputBias.size());

	grad.weightGradients.clear();
	grad.biasGradients.clear();

	for (size_t layer = 0; layer < m_params.weights.size(); ++layer)
	{
		grad.weightGradients.push_back(NNMatrix::Zero(m_params.weights[layer].rows(), m_params.weights[layer].cols()));
		grad.biasGradients.push_back(NNVector::Zero(1, m_params.weights[layer].cols()));
	}
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
template <typename Derived>
NNMatrixRM FCANN<ACTF, ACTFLast>::ForwardPropagate(const MatrixBase<Derived> &in, Activations &act)
{
	assert(act.act.size() == m_params.weights.size() + 1);
	assert(act.actIn.size() == m_params.weights.size() + 1);

	act.act[0] = in;
	act.actIn[0] = in; // first layer has no activation

	NNMatrixRM x = in;

	for (size_t layer = 0; layer < m_params.weights.size(); ++layer)
	{
		x *= m_params.weights[layer];

		x.rowwise() += m_params.outputBias[layer];

		act.actIn[layer + 1] = x;

		Activate_(x, layer == (m_params.weights.size() - 1));

		act.act[layer + 1] = x;
	}

	return x;
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
template <typename Derived>
NNMatrixRM FCANN<ACTF, ACTFLast>::ForwardPropagateFast(const MatrixBase<Derived> &in)
{
	/*
	if (!m_params.weightsSemiSparseCurrent)
	{
		UpdateWeightSemiSparse_();
	}
	*/

	for (size_t layer = 0; layer < m_params.weights.size(); ++layer)
	{
		if (layer == 0)
		{
			m_params.evalTmp[layer].noalias() = in * m_params.weights[layer];
			//MatrixMultiplyWithSemiSparse(in, m_params.weightsSemiSparse[layer], m_params.evalTmp[layer]);
		}
		else
		{
			m_params.evalTmp[layer].noalias() = m_params.evalTmp[layer - 1] * m_params.weights[layer];
			//MatrixMultiplyWithSemiSparse(m_params.evalTmp[layer - 1], m_params.weightsSemiSparse[layer], m_params.evalTmp[layer]);
		}

		m_params.evalTmp[layer].rowwise() += m_params.outputBias[layer];

		Activate_(m_params.evalTmp[layer], layer == (m_params.weights.size() - 1));
	}

	return m_params.evalTmp[m_params.weights.size() - 1];
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
template <typename Derived>
float FCANN<ACTF, ACTFLast>::ForwardPropagateSingle(const MatrixBase<Derived> &vec)
{
	if (!m_params.weightsSemiSparseCurrent)
	{
		UpdateWeightSemiSparse_();
	}

	for (size_t layer = 0; layer < m_params.weights.size(); ++layer)
	{
		if (layer == 0)
		{
			//m_params.evalSingleTmp[layer].noalias() = vec * m_params.weights[layer];
			MultiplyWithSemiSparse(vec, m_params.weightsSemiSparse[layer], m_params.evalSingleTmp[layer]);
		}
		else
		{
			//m_params.evalSingleTmp[layer].noalias() = m_params.evalSingleTmp[layer - 1] * m_params.weights[layer];
			MultiplyWithSemiSparse(m_params.evalSingleTmp[layer - 1], m_params.weightsSemiSparse[layer], m_params.evalSingleTmp[layer]);
		}

		m_params.evalSingleTmp[layer] += m_params.outputBias[layer];

		Activate_(m_params.evalSingleTmp[layer], layer == (m_params.weights.size() - 1));
	}

	return m_params.evalSingleTmp[m_params.weights.size() - 1](0, 0);
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
template <typename Derived>
float FCANN<ACTF, ACTFLast>::ForwardPropagateSingleWithSignature(const MatrixBase<Derived> &vec, float *signOut)
{
	if (!m_params.weightsSemiSparseCurrent)
	{
		UpdateWeightSemiSparse_();
	}

	for (size_t layer = 0; layer < m_params.weights.size(); ++layer)
	{
		if (layer == 0)
		{
			MultiplyWithSemiSparse(vec, m_params.weightsSemiSparse[layer], m_params.evalSingleTmp[layer]);
		}
		else
		{
			MultiplyWithSemiSparse(m_params.evalSingleTmp[layer - 1], m_params.weightsSemiSparse[layer], m_params.evalSingleTmp[layer]);
		}

		m_params.evalSingleTmp[layer] += m_params.outputBias[layer];

		Activate_(m_params.evalSingleTmp[layer], layer == (m_params.weights.size() - 1));

		if (layer == (m_params.weights.size() - 2))
		{
			size_t signatureSize = m_params.weights[layer].cols();

			for (size_t i = 0; i < signatureSize; ++i)
			{
				signOut[i] = m_params.evalSingleTmp[layer](0, i);
			}
		}
	}

	return m_params.evalSingleTmp[m_params.weights.size() - 1](0, 0);
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
template <typename Derived>
void FCANN<ACTF, ACTFLast>::BackwardPropagateComputeGrad(const MatrixBase<Derived> &err, const Activations &act, Gradients &grad)
{
	assert(grad.weightGradients.size() == m_params.weights.size());
	assert(grad.biasGradients.size() == m_params.outputBias.size());
	assert(grad.weightGradients.size() == grad.biasGradients.size());

	// currError are the errorTerms of the next layer
	NNMatrixRM errorTerms = err;

	for (int32_t layer = (m_params.weights.size() - 1); layer >= 0; --layer)
	{
		assert(grad.weightGradients[layer].rows() == m_params.weights[layer].rows());
		assert(grad.weightGradients[layer].cols() == m_params.weights[layer].cols());
		assert(grad.biasGradients[layer].rows() == 1);
		assert(grad.biasGradients[layer].cols() == m_params.outputBias[layer].cols());

		// first we calculate weight gradients for the current layer,
		// which is the transpose of each input to this layer, multiplied
		// by errorTerms
		grad.weightGradients[layer].noalias() = act.act[layer].transpose() * errorTerms;

		// bias gradients are just errorTerms
		grad.biasGradients[layer].noalias() = errorTerms.colwise().sum();

		NNMatrixRM derivatives = act.actIn[layer];
		ActivateDerivative_(derivatives);

		// then we calculate error for the next (previous) layer
		errorTerms *= m_params.weights[layer].transpose();
		errorTerms.array() *= derivatives.array();
	}
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
template <typename Derived1, typename Derived2>
float FCANN<ACTF, ACTFLast>::TrainGDM(const MatrixBase<Derived1> &x, const MatrixBase<Derived2> &y, float learningRate, float reg)
{
	static std::vector<Gradients> gradLocal;
	static std::vector<Activations> actLocal;
	static bool initialized = false;

	// we limit to 8 threads for the current block size of 256
	ScopedThreadLimiter tlim(8);

	if (!initialized)
	{
		gradLocal = std::vector<Gradients>(omp_get_max_threads());
		actLocal = std::vector<Activations>(omp_get_max_threads());

		for (int64_t i = 0; i < omp_get_max_threads(); ++i)
		{
			InitializeActivations(actLocal[i]);

			InitializeGradients(gradLocal[i]);
		}

		initialized = true;
	}

	float errorsMeasureTotal = 0.0f;

	#pragma omp parallel
	{
		int64_t begin;
		int64_t numRows;

		GetThreadBlock_(x.rows(), begin, numRows);

		size_t threadId = omp_get_thread_num();
		size_t numThreads = omp_get_num_threads();

		auto pred = ForwardPropagate(x.block(begin, 0, numRows, x.cols()), actLocal[threadId]);

		errorsMeasureTotal += ErrorFunc(pred, y.block(begin, 0, numRows, y.cols())).sum();

		NNMatrixRM errorsDerivative = ErrorFuncDerivative(pred, y.block(begin, 0, numRows, y.cols()), actLocal[threadId].actIn[actLocal[threadId].actIn.size() - 1]);

		BackwardPropagateComputeGrad(errorsDerivative, actLocal[threadId], gradLocal[threadId]);

		// reduce all the local gradients into total, using log(n) steps
		for (size_t skip = 2; skip <= numThreads; skip *= 2)
		{
			if ((threadId % skip) == 0 && (threadId + skip / 2) < numThreads)
			{
				gradLocal[threadId] += gradLocal[threadId + skip / 2];
			}
			#pragma omp barrier
		}
	}

	ApplyWeightUpdates(gradLocal[0], learningRate, reg);

	return errorsMeasureTotal / x.rows();
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
void FCANN<ACTF, ACTFLast>::ApplyWeightUpdates(const Gradients &grad, float learningRate, float reg)
{
	assert(grad.weightGradients.size() == m_params.weights.size());
	assert(grad.biasGradients.size() == m_params.outputBias.size());
	assert(grad.weightGradients.size() == grad.biasGradients.size());

	m_params.weightsLastUpdate.resize(m_params.weights.size());
	m_params.outputBiasLastUpdate.resize(m_params.outputBias.size());

	m_params.weightsEg2.resize(m_params.weights.size());
	m_params.outputBiasEg2.resize(m_params.outputBias.size());

	m_params.weightsRMSd2.resize(m_params.weights.size());
	m_params.outputBiasRMSd2.resize(m_params.outputBias.size());

	for (size_t layer = 0; layer < m_params.weights.size(); ++layer)
	{
		#pragma omp parallel
		{
			int64_t begin;
			int64_t numCols;

			size_t inSize = m_params.weights[layer].rows();
			size_t outSize = m_params.weights[layer].cols();

			GetThreadBlock_(outSize, begin, numCols);

			if (numCols != 0) // if numCols is less than num threads, some threads won't have anything to do
			{
				auto weightsBlock = m_params.weights[layer].block(0, begin, inSize, numCols);
				auto biasBlock = m_params.outputBias[layer].block(0, begin, 1, numCols);

				auto weightsGradientsBlock = grad.weightGradients[layer].block(0, begin, inSize, numCols);
				auto biasGradientsBlock = grad.biasGradients[layer].block(0, begin, 1, numCols);

				auto weightsEg2Block = m_params.weightsEg2[layer].block(0, begin, inSize, numCols);
				auto biasEg2Block = m_params.outputBiasEg2[layer].block(0, begin, 1, numCols);

				auto weightsRMSd2Block = m_params.weightsRMSd2[layer].block(0, begin, inSize, numCols);
				auto biasRMSd2Block = m_params.outputBiasRMSd2[layer].block(0, begin, 1, numCols);

				auto weightMaskBlock = m_params.weightMasks[layer].block(0, begin, inSize, numCols);

				#define L1_REG
				#ifdef L1_REG
				NNMatrix weightReg(weightsBlock.rows(), weightsBlock.cols());

				for (int64_t i = 0; i < (weightReg.rows() * weightReg.cols()); ++i)
				{
					float w = weightsBlock.data()[i];
					float x;

					if (w > 0.0f)
					{
						if (w > reg)
						{
							x = -reg;
						}
						else
						{
							x = -w;
						}
					}
					else
					{
						if (w < -reg)
						{
							x = reg;
						}
						else
						{
							x = -w;
						}
					}

					weightReg.data()[i] = x;
				}
				#elif defined(L2_REG)
				NNMatrix weightReg =  -reg * weightsBlock;
				#else
				NNMatrix weightReg = NNMatrix::Zero(weightsBlock.rows(), weightsBlock.cols());
				#endif

				// update Eg2 (ADADELTA)
				float decay = 0.99f;
				float e = 1e-8f;
				weightsEg2Block.array() *= decay;
				weightsEg2Block.array() += (weightsGradientsBlock.array() * weightsGradientsBlock.array()) * (1.0f - decay);
				biasEg2Block.array() *= decay;
				biasEg2Block.array() += (biasGradientsBlock.array() * biasGradientsBlock.array()) * (1.0f - decay);

				// ADADELTA
				NNMatrix weightDelta = -weightsGradientsBlock.array() * (weightsRMSd2Block.array() + e).sqrt() / (weightsEg2Block.array() + e).sqrt() + weightReg.array();
				NNVector biasDelta = -biasGradientsBlock.array() * (biasRMSd2Block.array() + e).sqrt() / (biasEg2Block.array() + e).sqrt();

				//NNMatrix weightDelta = -weightsGradientsBlock.array() * learningRate /*+ weightReg.array()*/;
				//NNVector biasDelta = -biasGradientsBlock.array() * learningRate;

				weightsBlock += weightDelta * learningRate;
				weightsBlock.array() *= weightMaskBlock.array();
				biasBlock += biasDelta * learningRate;

				FP weightMax = std::max(std::max(weightsBlock.maxCoeff(), -weightsBlock.minCoeff()), std::max(biasBlock.maxCoeff(), -biasBlock.minCoeff()));
				if (weightMax > MAX_WEIGHT)
				{
					throw LearningRateException();
				}

				// ADADELTA
				weightsRMSd2Block *= decay;
				weightsRMSd2Block.array() += weightDelta.array() * weightDelta.array() * (1.0f - decay);
				biasRMSd2Block *= decay;
				biasRMSd2Block.array() += biasDelta.array() * biasDelta.array() * (1.0f - decay);
			}

		} // parallel
	}

	m_params.weightsSemiSparseCurrent = false;
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
float FCANN<ACTF, ACTFLast>::GetSparsity()
{
	uint64_t zCount = 0;
	uint64_t totalCount = 0;

	for (size_t layer = 0; layer < m_params.weights.size(); ++layer)
	{
		totalCount += m_params.weights[layer].rows() * m_params.weights[layer].cols();

		for (int64_t i = 0; i < m_params.weights[layer].size(); ++i)
		{
			if (m_params.weights[layer].data()[i] == 0.0f)
			{
				++zCount;
			}
		}
	}

	return static_cast<float>(zCount) / totalCount;
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
template <typename Derived1, typename Derived2>
NNMatrixRM FCANN<ACTF, ACTFLast>::ErrorFunc(
	const MatrixBase<Derived1> &pred,
	const MatrixBase<Derived2> &targets) const
{
	NNMatrixRM ret;

	// for linear and tanh output we use MAE
	if (ACTFLast == Linear)
	{
		ret = (pred - targets).array().abs().matrix();
	}
	else if (ACTFLast == Tanh)
	{
		//ret = (pred - targets).array().abs().matrix();
		ret = ((pred - targets).array() * (pred - targets).array()).matrix();
	}
	// for softmax output we use cross-entropy
	else if (ACTFLast == Softmax)
	{
		// note that for cross-entropy, output is a vector no matter how many classes we have
		ret.resize(pred.rows(), 1);

		for (int64_t i = 0; i < pred.rows(); ++i)
		{
			float e = 0.0f;

			for (int64_t j = 0; j < pred.cols(); ++j)
			{
				if (targets(i, j) == 1.0f)
				{
					e += -log(pred(i, j));
				}
			}

			ret(i, 0) = e;
		}
	}
	else if (ACTFLast == Logsig)
	{
		// cross-entropy
		// - target * log(p) - (1 - target) * log(1-p)

		ret.resize(pred.rows(), pred.cols());

		for (int64_t i = 0; i < pred.rows(); ++i)
		{
			for (int64_t j = 0; j < pred.cols(); ++j)
			{
				ret(i, j) = -targets(i, j) * log(pred(i, j)) -(1.0f - targets(i, j)) * log(1.0f - pred(i, j));
			}
		}
	}
	else assert(false);

	return ret;
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
template <typename Derived1, typename Derived2, typename Derived3>
NNMatrixRM FCANN<ACTF, ACTFLast>::ErrorFuncDerivative(
	const MatrixBase<Derived1> &pred,
	const MatrixBase<Derived2> &targets,
	const MatrixBase<Derived3> &finalLayerActivations) const
{
	NNMatrixRM ret;

	if (ACTFLast == Linear)
	{
		NNMatrixRM err = pred - targets;

		ret.resize(err.rows(), err.cols());

		// MAE
		for (int64_t i = 0; i < err.rows(); ++i)
		{
			for (int64_t j = 0; j < err.cols(); ++j)
			{
				ret(i, j) = (err(i, j) > 0.0f) ? 1.0f : -1.0f;
			}
		}
	}
	else if (ACTFLast == Tanh)
	{
		ret = pred - targets;

		// now we have to multiply every element by the derivative at that point
		// derivative of tanh is 1-tanh^2(x)
		for (int64_t i = 0; i < ret.rows(); ++i)
		{
			for (int64_t j = 0; j < ret.cols(); ++j)
			{
				float tanhx = tanh(finalLayerActivations(i, j));
				ret(i, j) *= 1 - (tanhx * tanhx);
			}
		}
	}
	else if (ACTFLast == Softmax)
	{
		// cross-entropy
		// curiously,
		// http://www.willamette.edu/~gorr/classes/cs449/classify.html
		ret = pred - targets;
	}
	else if (ACTFLast == Logsig)
	{
		// with cross-entropy
		// http://cs229.stanford.edu/notes/cs229-notes1.pdf
		ret = pred - targets;
	}
	else assert(false);

	return ret;
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
template <typename Derived>
void FCANN<ACTF, ACTFLast>::Activate_(MatrixBase<Derived> &x, bool last) const
{
	ActivationFunc actf = last ? ACTFLast : ACTF;

	// these will all be optimized to just be a single case, since
	// ACTF is a template parameter
	if (actf == Linear)
	{
		return; // nothing to do here
	}
	else if (actf == Tanh)
	{
		for (int32_t i = 0; i < x.cols(); ++i)
		{
			for (int32_t j = 0; j < x.rows(); ++j)
			{
				x(j, i) = tanh(x(j, i));
			}
		}
	}
	else if (actf == Relu)
	{
		x = x.array().max(NNMatrix::Zero(x.rows(), x.cols()).array());
	}
	else if (actf == Softmax)
	{
		// the naive implementation is likely to overflow, so we do some shifting first
		// since we are in log space, and dividing in x is subtracting in log(x)
		// dividing all values won't change the distribution

		// we find the max component in each row, and subtract that from each component
		Eigen::Matrix<FP, Eigen::Dynamic, 1> maxElems = x.rowwise().maxCoeff();

		x.colwise() -= maxElems;

		// compute element-wise exp
		x = x.array().exp().matrix();

		// then compute the normalization denominator for each row
		Eigen::Matrix<FP, Eigen::Dynamic, 1> norm = x.rowwise().sum();

		// then normalize all the elements
		x.array().colwise() /= norm.array();
	}
	else if (actf == Logsig)
	{
		// 1 / (exp(-x) + 1)
		x = (1.0f / ((-x).array().exp() + 1)).matrix();
	}
	else assert(false);
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
template <typename Derived>
void FCANN<ACTF, ACTFLast>::ActivateDerivative_(MatrixBase<Derived> &x) const
{
	// these will all be optimized to just be a single case, since
	// ACTF is a template parameter
	if (ACTF == Linear)
	{
		x = NNMatrixRM::Ones(x.rows(), x.cols());
	}
	else if (ACTF == Tanh)
	{
		// derivative of tanh is 1-tanh^2(x)

		for (int32_t i = 0; i < x.cols(); ++i)
		{
			for (int32_t j = 0; j < x.rows(); ++j)
			{
				FP tanhx = tanh(x(j, i));
				x(j, i) = 1 - (tanhx * tanhx);
			}
		}
	}
	else if (ACTF == Relu)
	{
		x.array() = (x.array() > NNMatrixRM::Zero(x.rows(), x.cols()).array());
	}
	else assert(false);
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
void FCANN<ACTF, ACTFLast>::UpdateWeightMasksRegions_()
{
	m_params.weightMasksRegions.resize(m_params.weightMasks.size());

	for (size_t layer = 0; layer < m_params.weightMasks.size(); ++layer)
	{
		WeightMaskType toConvert = m_params.weightMasks[layer];

		m_params.weightMasksRegions[layer] = MatrixToRegions(toConvert);

		int64_t totalSize = 0;
		for (const auto &region : m_params.weightMasksRegions[layer])
		{
			totalSize += region.rows * region.cols;
		}
	}

	m_params.weightsSemiSparseCurrent = false;
}

template <ActivationFunc ACTF, ActivationFunc ACTFLast>
void FCANN<ACTF, ACTFLast>::UpdateWeightSemiSparse_()
{
	m_params.weightsSemiSparse.resize(m_params.weightMasks.size());

	for (size_t layer = 0; layer < m_params.weightMasks.size(); ++layer)
	{
		WeightType toConvert = m_params.weights[layer];

		m_params.weightsSemiSparse[layer] = ToSemiSparse(toConvert, m_params.weightMasksRegions[layer]);
	}

	m_params.weightsSemiSparseCurrent = true;
}

/* serialization format:
 * numLayers
 * for each layer:
 *		weight matrix
 *		weight mask
 *		bias
 *
 * For each matrix:
 *	rows
 *	cols
 *  each field in row major format (rows * cols)
 */

namespace
{

template <typename Derived>
void PushMatrix(Eigen::MatrixBase<Derived> &m, std::ostream &s)
{
	s << m.rows() << ' ' << '\n';
	s << m.cols() << ' ' << '\n';

	for (int64_t row = 0; row < m.rows(); ++row)
	{
		for (int64_t col = 0; col < m.cols(); ++col)
		{
			s << m(row, col) << ' ';
		}
		s << '\n';
	}
}

NNMatrix ReadMatrix(std::istream &s)
{
	int64_t nRows;
	int64_t nCols;

	s >> nRows;
	s >> nCols;

	NNMatrix ret(nRows, nCols);

	for (int64_t row = 0; row < nRows; ++row)
	{
		for (int64_t col = 0; col < nCols; ++col)
		{
			s >> ret(row, col);
		}
	}

	return ret;
}

}

template <typename T>
void SerializeNet(T &net, std::ostream &s)
{
	auto weights = net.Weights();
	auto biases = net.Biases();
	auto weightMasks = net.WeightMasks();

	int64_t numLayers = weights.size();

	std::vector<size_t> hiddenLayerSizes;

	for (int64_t i = 1; i < numLayers; ++i)
	{
		hiddenLayerSizes.push_back(weights[i].rows());
	}

	s << numLayers << '\n';

	for (int64_t i = 0; i < numLayers; ++i)
	{
		PushMatrix(weights[i], s);
		PushMatrix(weightMasks[i], s);
		PushMatrix(biases[i], s);
	}
}

template <typename T>
void DeserializeNet(T &net, std::istream &s)
{
	std::vector<typename T::WeightType> weights;
	std::vector<typename T::BiasType> biases;
	std::vector<typename T::WeightMaskType> weightMasks;

	int64_t numLayers;

	s >> numLayers;

	for (int64_t i = 0; i < numLayers; ++i)
	{
		weights.push_back(ReadMatrix(s));
		weightMasks.push_back(ReadMatrix(s));
		biases.push_back(ReadMatrix(s));
	}

	int64_t din = weights[0].rows();
	int64_t dout = weights[weights.size() - 1].cols();

	std::vector<size_t> hiddenLayerSizes;

	for (int64_t i = 1; i < numLayers; ++i)
	{
		hiddenLayerSizes.push_back(weights[i].rows());
	}

	// we just set everything to be fully connected, since we will
	// overwrite the connection matrices anyways
	std::vector<std::vector<Eigen::Triplet<FP> > > connections(hiddenLayerSizes.size() + 1);

	net = T(din, dout, hiddenLayerSizes, connections);

	net.Weights() = weights;
	net.Biases() = biases;
	net.WeightMasks() = weightMasks;

	net.NotifyWeightMasksChanged();
}
