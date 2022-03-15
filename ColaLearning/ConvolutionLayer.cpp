#include "ConvolutionLayer.h"


ConvolutionLayer::ConvolutionLayer(map<Tensor, double> node_Values, Tensor _filter, Tensor _stride, Tensor _padding,
	const ActivationFunction& activation_Function, bool _bias)
	: Layer(node_Values, LayerType::Convolution, activation_Function, _bias)
{
	filter = _filter;
	stride = _stride;
	padding = _padding;
}
