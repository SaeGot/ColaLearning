#pragma once
#include "Layer.h"


class PoolingLayer : public Layer
{
public:
	enum class PoolingType
	{
		Max
	};
	PoolingLayer(Tensor pool_Size, Tensor _stride = Tensor(1, 1), PoolingType pooling_Type = PoolingType::Max);
	PoolingLayer(const PoolingLayer& layer);
	
	Tensor GetPoolSize();
	Tensor GetStride();
	virtual void SetNodes(Layer previous_Layer);

protected:
	Tensor poolSize;
	Tensor stride;
	PoolingType poolingType;
};

