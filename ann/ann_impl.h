#include "ann.h"

#include <iostream>
#include <memory>
#include <algorithm>

#include <cstdint>

#include <omp.h>

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

	m_params.weightsGpuTmp.resize(hiddenLayers.size() + 1);
	m_params.weightsTransGpuTmp.resize(hiddenLayers.size() + 1);
	m_params.xGpuTmp.resize(hiddenLayers.size() + 2); // we also need tmp for result of final layer
	m_params.errorTermGpuTmp.resize(hiddenLayers.size() + 2);
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
NNMatrixRM FCANN<ACTF>::ForwardPropagate(const MatrixBase<Derived> &in, Activations &act)
{
	assert(act.act.size() == m_params.weights.size() + 1);
	assert(act.actIn.size() == m_params.weights.size() + 1);

	act.act[0] = in;
	act.actIn[0] = in; // first layer has no activation

	NNMatrixRM x = in;

	for (size_t layer = 0; layer < m_params.weights.size(); ++layer)
	{
#ifdef VIENNACL_WITH_OPENCL
		MultiplyGPU(x, m_params.weights[layer], m_params.xGpuTmp[layer], m_params.weightsGpuTmp[layer], m_params.xGpuTmp[layer + 1]);
#else
		x *= m_params.weights[layer];
#endif

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
NNMatrixRM FCANN<ACTF>::ForwardPropagateFast(const MatrixBase<Derived> &in)
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
float FCANN<ACTF>::ForwardPropagateSingle(const MatrixBase<Derived> &vec)
{
	NNVector x;

	for (size_t layer = 0; layer < m_params.weights.size(); ++layer)
	{
		if (layer == 0)
		{
			x.noalias() = vec * m_params.weights[layer];
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

	return x(0, 0);
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
#ifdef VIENNACL_WITH_OPENCL
		NNMatrixRM weightsTrans = m_params.weights[layer].transpose();
		MultiplyGPU(errorTerms, weightsTrans, m_params.errorTermGpuTmp[layer], m_params.weightsTransGpuTmp[layer], m_params.errorTermGpuTmp[layer + 1]);
#else
		errorTerms *= m_params.weights[layer].transpose();
#endif
		errorTerms.array() *= derivatives.array();
	}
}

template <ActivationFunc ACTF>
template <typename Derived>
float FCANN<ACTF>::TrainGDM(const MatrixBase<Derived> &x, const MatrixBase<Derived> &y, float reg)
{
	static std::vector<Gradients> gradLocal;
	static std::vector<Activations> actLocal;
	static bool initialized = false;

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

	NNMatrixRM errorsMeasureTotal = NNMatrixRM::Zero(x.rows(), y.cols());

	#pragma omp parallel
	{
		int64_t begin;
		int64_t numRows;

		GetThreadBlock_(x.rows(), begin, numRows);

		size_t threadId = omp_get_thread_num();
		size_t numThreads = omp_get_num_threads();

		auto pred = ForwardPropagate(x.block(begin, 0, numRows, x.cols()), actLocal[threadId]);

		// these are the errors for propagation
		NNMatrixRM errorsPropagate(pred.rows(), pred.cols());

		auto errorsMeasure = pred - y.block(begin, 0, numRows, y.cols());

		ErrorFuncDeri(errorsMeasure, errorsPropagate);

		BackwardPropagateComputeGrad(errorsPropagate, actLocal[threadId], gradLocal[threadId]);

		errorsMeasureTotal.block(begin, 0, numRows, errorsMeasureTotal.cols())
				= errorsMeasure;

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

	ApplyWeightUpdates(gradLocal[0], reg);

	NNMatrixRM errors = NNMatrixRM::Zero(errorsMeasureTotal.rows(), errorsMeasureTotal.cols());
	ErrorFunc(errorsMeasureTotal, errors);

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

				weightsBlock += weightDelta;
				weightsBlock.array() *= weightMaskBlock.array();
				biasBlock += biasDelta;

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
		x = NNMatrixRM::Ones(x.rows(), x.cols());
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
		x.array() = (x.array() > NNMatrixRM::Zero(x.rows(), x.cols()).array());
	}
	else assert(false);
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

void SerializeNet(ANN &net, std::ostream &s)
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

ANN DeserializeNet(std::istream &s)
{
	std::vector<typename ANN::WeightType> weights;
	std::vector<typename ANN::BiasType> biases;
	std::vector<typename ANN::WeightMaskType> weightMasks;

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

	ANN ret(42, din, dout, hiddenLayerSizes, connections);

	ret.Weights() = weights;
	ret.Biases() = biases;
	ret.WeightMasks() = weightMasks;

	return ret;
}
