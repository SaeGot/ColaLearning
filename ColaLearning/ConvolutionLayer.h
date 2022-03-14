#pragma once
#include "Layer.h"

class ConvolutionLayer : public Layer
{
public:
	ConvolutionLayer(map<Tensor, double> node_Values, Tensor _filter, Tensor _stride, Tensor _padding = Tensor(0, 0),
		const ActivationFunction& activation_Function = ActivationFunction::Linear, bool _bias = true);

protected:
	Tensor filter;
	Tensor stride;
	Tensor padding;
};

