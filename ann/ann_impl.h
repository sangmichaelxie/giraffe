#include "ann.h"

#include <iostream>
#include <memory>
#include <algorithm>

#include <cstdint>

// for floating point interrupts
#include <xmmintrin.h>

void EnableNanInterrupt()
{
	_MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
}

template <ActivationFunc ACTF>
FCANN<ACTF>::FCANN(
	int randomSeed,
	size_t inputs,
	size_t outputs,
	std::vector<size_t> hiddenLayers,
	std::vector<std::vector<Eigen::Triplet<FP> > > &connectionMatrices)
	: m_mersenneTwister(randomSeed)
{
	if (connectionMatrices.size() != (hiddenLayers.size() + 1))
	{
		throw std::runtime_error("connectionMatrices.size() should be hiddenLayers.size() + 1");
	}

	// then we build the weight and bias vectors, and initialize them
	for (size_t layer = 0; layer < (hiddenLayers.size() + 1); ++layer)
	{
		size_t in_size = (layer == 0) ? inputs : hiddenLayers[layer - 1];
		size_t out_size = (layer == hiddenLayers.size()) ? outputs : hiddenLayers[layer];

		// we do initialization in eigen types even when using OpenCL, because element-wise access
		// from GPU is very slow
		NNMatrix weightMatrix(in_size, out_size);
		NNVector biasVector(out_size);

		// determine what distribution to use to initialize hidden weights depending on activation func
		// (output layer is always linear)
		std::function<FP()> drawFunc;

		if (ACTF == Linear || layer == hiddenLayers.size())
		{
			std::uniform_real_distribution<FP> dist(-0.01f, 0.01f);
			drawFunc = std::bind(dist, m_mersenneTwister);
		}
		else if (ACTF == Tanh)
		{
			// for tanh, we use r = sqrt(6/(fan_in + fan_out)), (-r, r)
			FP r = sqrt(6.0/(in_size + out_size));
			std::uniform_real_distribution<FP> dist(-r, r);
			drawFunc = std::bind(dist, m_mersenneTwister);
		}
		else if (ACTF == Relu)
		{
			// we use the scheme described here - http://arxiv.org/pdf/1502.01852v1.pdf
			std::normal_distribution<FP> dist(0.0f, sqrt(2.0f/out_size));
			drawFunc = std::bind(dist, m_mersenneTwister);
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

	SetNumThreads(1);
}

template <ActivationFunc ACTF>
FCANN<ACTF> &FCANN<ACTF>::operator=(const FCANN &other)
{
	this->m_params = other.m_params;
	SetNumThreads(other.m_computeThreads.size());

	return *this;
}

template <ActivationFunc ACTF>
void FCANN<ACTF>::InitializeActivations(Activations &act)
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

template <ActivationFunc ACTF>
void FCANN<ACTF>::InitializeGradients(Gradients &grad)
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

template <ActivationFunc ACTF>
template <typename Derived>
NNMatrix FCANN<ACTF>::ForwardPropagate(const MatrixBase<Derived> &in, Activations &act)
{
	assert(act.act.size() == m_params.weights.size() + 1);
	assert(act.actIn.size() == m_params.weights.size() + 1);

	act.act[0] = in;
	act.actIn[0] = in; // first layer has no activation

	NNMatrixRM x;

	for (size_t layer = 0; layer < m_params.weights.size(); ++layer)
	{
		if (layer == 0)
		{
			x.noalias() = in * m_params.weights[layer];
		}
		else
		{
			x *= m_params.weights[layer];
		}

		x.rowwise() += m_params.outputBias[layer];

		act.actIn[layer + 1] = x;

		if (layer != (m_params.weights.size() - 1))
		{
			Activate_(x);
		}

		act.act[layer + 1] = x;
	}

	return x;
}

template <ActivationFunc ACTF>
template <typename Derived>
NNMatrix FCANN<ACTF>::ForwardPropagateFast(const MatrixBase<Derived> &in)
{
	NNMatrixRM x;

	for (size_t layer = 0; layer < m_params.weights.size(); ++layer)
	{
		if (layer == 0)
		{
			x.noalias() = in * m_params.weights[layer];
		}
		else
		{
			x *= m_params.weights[layer];
		}

		x.rowwise() += m_params.outputBias[layer];

		if (layer != (m_params.weights.size() - 1))
		{
			Activate_(x);
		}
	}

	return x;
}

template <ActivationFunc ACTF>
template <typename Derived>
void FCANN<ACTF>::BackwardPropagateComputeGrad(const MatrixBase<Derived> &err, const Activations &act, Gradients &grad)
{
	assert(grad.weightGradients.size() == m_params.weights.size());
	assert(grad.biasGradients.size() == m_params.outputBias.size());
	assert(grad.weightGradients.size() == grad.biasGradients.size());

	// currError are the errorTerms of the next layer
	// for the output layer it's simply the network error (since activation is linear)
	NNMatrix errorTerms = err;

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

		NNMatrix derivatives = act.actIn[layer];
		ActivateDerivative_(derivatives);

		// then we calculate error for the next (previous) layer
		errorTerms *= m_params.weights[layer].transpose();
		errorTerms.array() *= derivatives.array();
	}
}

template <ActivationFunc ACTF>
float FCANN<ACTF>::TrainGDM(const NNMatrix &x, const NNMatrix &y, float reg)
{
	size_t numThreads = m_computeThreads.size();
	size_t rowsPerThread = x.rows() / numThreads;
	size_t rem = x.rows() % numThreads; // the first "rem" threads get 1 extra row
	int64_t begin = 0;

	for (size_t thread = 0; thread < numThreads; ++thread)
	{
		size_t numRowsForThisThread = rowsPerThread + ((thread < rem) ? 1 : 0);

		m_threadControls[thread]->x = &x;
		m_threadControls[thread]->y = &y;
		m_threadControls[thread]->rowBegin = begin;
		m_threadControls[thread]->numRows = numRowsForThisThread;

		begin += numRowsForThisThread;

		{
			std::lock_guard<std::mutex> lock(m_threadControls[thread]->mtx);
			m_threadControls[thread]->run = true;
		}
		m_threadControls[thread]->cv.notify_one();
	}

	assert(begin == x.rows());

	Gradients gradTotal;
	InitializeGradients(gradTotal);

	NNMatrix errorsMeasure = NNMatrix::Zero(x.rows(), y.cols());

	begin = 0;

	// now we wait till all threads are done, and accumulate their results
	for (size_t thread = 0; thread < numThreads; ++thread)
	{
		{
			std::unique_lock<std::mutex> lock(m_threadControls[thread]->mtx);
			m_threadControls[thread]->cv.wait(lock, [this, thread]{ return m_threadControls[thread]->run == false; });
		}

		gradTotal += m_threadControls[thread]->grad;

		errorsMeasure.block(begin, 0, m_threadControls[thread]->errorsMeasure.rows(), m_threadControls[thread]->errorsMeasure.cols())
				= m_threadControls[thread]->errorsMeasure;

		begin += m_threadControls[thread]->errorsMeasure.rows();
	}

	ApplyWeightUpdates(gradTotal, reg);

	NNMatrix errors = NNMatrix::Zero(errorsMeasure.rows(), errorsMeasure.cols());
	ErrorFunc(errorsMeasure, errors);

	return errors.sum() / x.rows();
}

template <ActivationFunc ACTF>
void FCANN<ACTF>::ApplyWeightUpdates(const Gradients &grad, float reg)
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
#define L1_REG
#ifdef L1_REG
		NNMatrix weightReg(m_params.weights[layer].rows(), m_params.weights[layer].cols());

		for (int64_t i = 0; i < (weightReg.rows() * weightReg.cols()); ++i)
		{
			float w = m_params.weights[layer].data()[i];
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
		NNMatrix weightReg =  -reg * m_params.weights[layer];

		NNMatrix weightDelta = -grad.weightGradients[layer] * learningRate + momentum * m_params.weightsLastUpdate[layer] + weightReg;
		NNVector biasDelta = -grad.biasGradients[layer] * learningRate + momentum * m_params.outputBiasLastUpdate[layer];
#else
		NNMatrix weightReg = NNMatrix::Zero(m_params.weights[layer].rows(), m_params.weights[layer].cols());
#endif

		// update Eg2 (ADADELTA)
		float decay = 0.99f;
		float e = 1e-8f;
		m_params.weightsEg2[layer].array() *= decay;
		m_params.weightsEg2[layer].array() += (grad.weightGradients[layer].array() * grad.weightGradients[layer].array()) * (1.0f - decay);
		m_params.outputBiasEg2[layer].array() *= decay;
		m_params.outputBiasEg2[layer].array() += (grad.biasGradients[layer].array() * grad.biasGradients[layer].array()) * (1.0f - decay);

		// ADADELTA
		NNMatrix weightDelta = -grad.weightGradients[layer].array() * (m_params.weightsRMSd2[layer].array() + e).sqrt() / (m_params.weightsEg2[layer].array() + e).sqrt() + weightReg.array();
		NNVector biasDelta = -grad.biasGradients[layer].array() * (m_params.outputBiasRMSd2[layer].array() + e).sqrt() / (m_params.outputBiasEg2[layer].array() + e).sqrt();

		m_params.weights[layer] += weightDelta;
		m_params.weights[layer].array() *= m_params.weightMasks[layer].array();
		m_params.outputBias[layer] += biasDelta;

		// SGD Nesterov Momentum
//		NNMatrix weightsV = -grad.weightGradients[layer] * learningRate + momentum * m_params.weightsLastUpdate[layer];
//		NNVector biasV = -grad.biasGradients[layer] * learningRate + momentum * m_params.outputBiasLastUpdate[layer];

//		m_params.weights[layer] += momentum * weightsV - learningRate * grad.weightGradients[layer];
//		m_params.outputBias[layer] += momentum * biasV - learningRate * grad.biasGradients[layer];

		FP weightMax = std::max(std::max(m_params.weights[layer].maxCoeff(), -m_params.weights[layer].minCoeff()), std::max(m_params.outputBias[layer].maxCoeff(), -m_params.outputBias[layer].minCoeff()));
		if (weightMax > MAX_WEIGHT)
		{
			throw LearningRateException();
		}

		// ADADELTA
		m_params.weightsRMSd2[layer] *= decay;
		m_params.weightsRMSd2[layer].array() += weightDelta.array() * weightDelta.array() * (1.0f - decay);
		m_params.outputBiasRMSd2[layer] *= decay;
		m_params.outputBiasRMSd2[layer].array() += biasDelta.array() * biasDelta.array() * (1.0f - decay);

		// SGD Momentum
//		m_params.weightsLastUpdate[layer] = weightsV;
//		m_params.outputBiasLastUpdate[layer] = biasV;
	}
}

template <ActivationFunc ACTF>
void FCANN<ACTF>::SetNumThreads(size_t n)
{
	if (n == m_computeThreads.size())
	{
		return;
	}
	else if (n < m_computeThreads.size())
	{
		// we are downsizing
		for (size_t i = n; i < m_computeThreads.size(); ++i)
		{
			// signal each extra thread to exit, and join them
			{
				std::lock_guard<std::mutex> lock(m_threadControls[i]->mtx);
				m_threadControls[i]->exit = true;
			}
			m_threadControls[i]->cv.notify_one();

			m_computeThreads[i].join();
		}

		// finally reduce the size of vectors
		m_threadControls.resize(n);
		m_computeThreads.resize(n);
	}
	else
	{
		// we are upsizing!
		size_t originalSize = m_computeThreads.size();

		for (size_t i = originalSize; i < n; ++i)
		{
			m_threadControls.push_back(std::unique_ptr<ComputeThreadControl>(new ComputeThreadControl));
			m_computeThreads.push_back(std::move(std::thread(&FCANN::ComputeThreadMain_, this, &(*m_threadControls[i]))));
		}
	}
}

template <ActivationFunc ACTF>
float FCANN<ACTF>::GetSparsity()
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

template <ActivationFunc ACTF>
template <typename Derived>
void FCANN<ACTF>::Activate_(MatrixBase<Derived> &x) const
{
	// these will all be optimized to just be a single case, since
	// ACTF is a template parameter
	if (ACTF == Linear)
	{
		return; // nothing to do here
	}
	else if (ACTF == Tanh)
	{
		for (int32_t i = 0; i < x.cols(); ++i)
		{
			for (int32_t j = 0; j < x.rows(); ++j)
			{
				x(j, i) = tanh(x(j, i));
			}
		}
	}
	else if (ACTF == Relu)
	{
		x = x.array().max(NNMatrix::Zero(x.rows(), x.cols()).array());
	}
	else assert(false);
}

template <ActivationFunc ACTF>
template <typename Derived>
void FCANN<ACTF>::ActivateDerivative_(MatrixBase<Derived> &x) const
{
	// these will all be optimized to just be a single case, since
	// ACTF is a template parameter
	if (ACTF == Linear)
	{
		x = NNMatrix::Ones(x.rows(), x.cols());
	}
	else if (ACTF == Tanh)
	{
		for (int32_t i = 0; i < x.cols(); ++i)
		{
			for (int32_t j = 0; j < x.rows(); ++j)
			{
				FP coshx = cosh(x(j, i));
				x(j, i) = (fabs(x(j, i)) > 20.0f) ? 0.0f : 1/(coshx * coshx);
			}
		}
	}
	else if (ACTF == Relu)
	{
		x.array() = (x.array() > NNMatrix::Zero(x.rows(), x.cols()).array());
	}
	else assert(false);
}

template <ActivationFunc ACTF>
void FCANN<ACTF>::ComputeThreadMain_(ComputeThreadControl *ctrl)
{
	Activations act;
	InitializeActivations(act);
	InitializeGradients(ctrl->grad);

	while (!ctrl->exit)
	{
		std::unique_lock<std::mutex> lock(ctrl->mtx);
		ctrl->cv.wait(lock, [&ctrl]{ return ctrl->exit || ctrl->run; });

		if (ctrl->exit)
		{
			lock.unlock();
			break;
		}

		NNMatrix pred = ForwardPropagate(ctrl->x->block(ctrl->rowBegin, 0, ctrl->numRows, ctrl->x->cols()), act);

		// these are the errors for propagation
		NNMatrix errorsPropagate(pred.rows(), pred.cols());

		ctrl->errorsMeasure = pred - ctrl->y->block(ctrl->rowBegin, 0, ctrl->numRows, ctrl->y->cols());

		ErrorFuncDeri(ctrl->errorsMeasure, errorsPropagate);

		BackwardPropagateComputeGrad(errorsPropagate, act, ctrl->grad);

		ctrl->run = false;

		lock.unlock();
		ctrl->cv.notify_one();
	}
}
