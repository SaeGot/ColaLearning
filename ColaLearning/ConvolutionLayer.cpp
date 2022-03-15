#include "ConvolutionLayer.h"


ConvolutionLayer::ConvolutionLayer(map<Tensor, double> node_Values, Tensor _filter,
	const ActivationFunction& activation_Function, bool _bias,
	Tensor _stride, Tensor _padding)
	: Layer(node_Values, LayerType::Convolution, activation_Function, _bias)
{
	filter = _filter + Tensor(0, 0, 1);
	stride = _stride;
	padding = _padding;
	channel = 0;
	for (const pair<Tensor, double>& node_values : node_Values)
	{
		if (node_values.first.GetXYChannelSize()[2] > channel)
		{
			channel = node_values.first.GetXYChannelSize()[2];
		}
	}
}

ConvolutionLayer::ConvolutionLayer(int _channel, Tensor _filter,
	const ActivationFunction& activation_Function, bool _bias,
	Tensor _stride, Tensor _padding)
{
	layerType = LayerType::Convolution;
	activationFunction = activation_Function;
	bias = _bias;
	filter = _filter + Tensor(0, 0, 1);
	stride = _stride;
	padding = _padding;
	channel = _channel;
}

ConvolutionLayer::ConvolutionLayer(const ConvolutionLayer& layer) : Layer (layer)
{
	filter = layer.filter;
	stride = layer.stride;
	padding = layer.padding;
	channel = layer.channel;
}

Tensor ConvolutionLayer::GetFilter()
{
	return filter;
}

Tensor ConvolutionLayer::GetStride()
{
	return stride;
}

Tensor ConvolutionLayer::GetPadding()
{
	return padding;
}

void ConvolutionLayer::SetNodes(Layer previous_Layer)
{
	vector<int> previous_layer_size = previous_Layer.GetLayerSize();
	vector<int> padding_size = padding.GetXYChannelSize();
	vector<int> stride_size = stride.GetXYChannelSize();
	vector<int> filter_size = filter.GetXYChannelSize();
	vector<int> layer_size;
	for (int n = 0; n < 2; n++)
	{
		layer_size.push_back(1 + (previous_layer_size[n] + (2 * padding_size[n] - filter_size[n])) / stride_size[n]);
	}
	layerSize = Tensor(layer_size[0], layer_size[1], channel, bias);
	vector<int> size = layerSize.GetXYChannel();
	for (int i = 0; i < size[0]; i++)
	{
		for (int j = 0; j < size[1]; j++)
		{
			for (int k = 0; k < size[2]; k++)
			{
				nodeValues.emplace(Tensor(i, j, k), 0);
				backNodeValues.emplace(Tensor(i, j, k), 0);
			}
		}
	}
}
