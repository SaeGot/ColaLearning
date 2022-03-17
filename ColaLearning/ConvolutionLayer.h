#pragma once
#include "Layer.h"

class ConvolutionLayer : public Layer
{
public:
	ConvolutionLayer(map<Tensor, double> node_Values, Tensor _filter,
		const ActivationFunction& activation_Function = ActivationFunction::Linear, bool _bias = true,
		Tensor _stride = Tensor(1, 1), Tensor _padding = Tensor(0, 0));
	ConvolutionLayer(int _channel, Tensor _filter,
		const ActivationFunction& activation_Function = ActivationFunction::Linear, bool _bias = true,
		Tensor _stride = Tensor(1, 1), Tensor _padding = Tensor(0, 0));
	ConvolutionLayer(const ConvolutionLayer& layer);
	
	Tensor GetFilter();
	Tensor GetStride();
	Tensor GetPadding();
	virtual void SetNodes(Layer previous_Layer);
	void AddWeightConnection(Tensor next_Tensor, vector<Tensor> prev_Tensors, vector<Tensor> filter_Tensors);
	vector<Tensor> GetPreviousConnectionWithoutBias(Tensor next_Tensor) const;
	vector<Tensor> GetFilterConnectionWithoutBias(Tensor next_Tensor) const;

protected:
	Tensor filter;
	Tensor stride;
	Tensor padding;
	int channel;
	// 다음층과 연결된 이전층
	map<Tensor, vector<Tensor>> weightConnectionWithoutBias;
	// 다음층과 연결된 필터
	map<Tensor, vector<Tensor>> filterConnectionWithoutBias;
};

