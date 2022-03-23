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
	int pool_x = poolSize.GetXYChannel()[0];
	int pool_y = poolSize.GetXYChannel()[1];
	int stride_x = stride.GetXYChannel()[0];
	int stride_y = stride.GetXYChannel()[1];
	int node_count_x = 1 + floorl( (prev_node_count_x - pool_x) / stride_x );
	int node_count_y = 1 + floorl((prev_node_count_y - pool_y) / stride_y);
	layerSize = Tensor(node_count_x, node_count_y, 0, bias);
	for (int i = 0; i < node_count_x; i++)
	{
		for (int j = 0; j < node_count_y; j++)
		{
			nodeValues.emplace(Tensor(i, j), 0);
			backNodeValues.emplace(Tensor(i, j), 0);
		}
	}
}