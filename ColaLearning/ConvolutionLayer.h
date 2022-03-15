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

protected:
	Tensor filter;
	Tensor stride;
	Tensor padding;
	// 채널 개수
	int channel;
};

