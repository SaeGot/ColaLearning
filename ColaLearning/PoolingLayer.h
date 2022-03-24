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
	
	vector<Tensor> GetPools();
	Tensor GetPoolSize();
	Tensor GetStride();
	virtual void SetNodes(Layer previous_Layer);
	void AddWeightConnection(Tensor next_Tensor, vector<Tensor> prev_Tensors);
	vector<Tensor> GetPreviousConnectionWithoutBias(Tensor next_Tensor) const;
	PoolingType GetPoolingType();

protected:
	Tensor poolSize;
	Tensor stride;
	PoolingType poolingType;
	// 다음층과 연결된 이전층
	map<Tensor, vector<Tensor>> weightConnectionWithoutBias;
};

