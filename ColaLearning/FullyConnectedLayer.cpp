#include "FullyConnectedLayer.h"


FullyConnectedLayer::~FullyConnectedLayer()
{
	Layer::Initialize();
}
/*
double FullyConnectedLayer::ForwardSum(const Weight& weight, Tensor j)
{
	double output_sum = 0.0;
	for (const Tensor& tensor : weight.GetJConnectedWeightTensorWithoutBias(j))
	{
		output_sum += GetNodeValue(tensor) * weight.GetWeight(tensor, j);
	}
	if (CheckBias())
	{
		output_sum += weight.GetWeight(Tensor::GetBias(), j);
	}

	return output_sum;
}
*/