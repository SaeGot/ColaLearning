#include "PoolingLayer.h"


PoolingLayer::PoolingLayer(Tensor pool_Size, Tensor _stride, PoolingType pooling_Type)
{
	layerType = LayerType::Pooling;
	poolSize = pool_Size;
	stride = _stride;
	poolingType = pooling_Type;

	activationFunction = ActivationFunction::Linear;
	bias = false;
}

PoolingLayer::PoolingLayer(const PoolingLayer& layer) : Layer(layer)
{
	layerType = LayerType::Pooling;
	poolSize = layer.poolSize;
	stride = layer.stride;
	poolingType = layer.poolingType;
}

vector<Tensor> PoolingLayer::GetPools()
{
	int x = poolSize.GetXYChannel()[0];
	int y = poolSize.GetXYChannel()[0];
	vector<Tensor> tensors;
	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < y; j++)
		{
			tensors.push_back(Tensor(i, j));
		}
	}

	return tensors;
}

Tensor PoolingLayer::GetPoolSize()
{
	return poolSize;
}

Tensor PoolingLayer::GetStride()
{
	return stride;
}

void PoolingLayer::SetNodes(Layer previous_Layer)
{
	int prev_node_count_x = previous_Layer.GetLayerSize()[0];
	int prev_node_count_y = previous_Layer.GetLayerSize()[1];
	int prev_channel = previous_Layer.GetLayerSize()[2];
	int pool_x = poolSize.GetXYChannel()[0];
	int pool_y = poolSize.GetXYChannel()[1];
	int stride_x = stride.GetXYChannel()[0];
	int stride_y = stride.GetXYChannel()[1];
	int node_count_x = 1 + floorl( (prev_node_count_x - pool_x) / stride_x );
	int node_count_y = 1 + floorl((prev_node_count_y - pool_y) / stride_y);
	bias = previous_Layer.CheckBias();
	layerSize = Tensor(node_count_x, node_count_y, prev_channel, bias);
	for (int i = 0; i < node_count_x; i++)
	{
		for (int j = 0; j < node_count_y; j++)
		{
			nodeValues.emplace(Tensor(i, j), 0);
			backNodeValues.emplace(Tensor(i, j), 0);
		}
	}
}

void PoolingLayer::AddWeightConnection(Tensor next_Tensor, vector<Tensor> prev_Tensors)
{
	weightConnectionWithoutBias.emplace(next_Tensor, prev_Tensors);
}

vector<Tensor> PoolingLayer::GetPreviousConnectionWithoutBias(Tensor next_Tensor) const
{
	return weightConnectionWithoutBias.at(next_Tensor);
}

PoolingLayer::PoolingType PoolingLayer::GetPoolingType()
{
	return poolingType;
}
