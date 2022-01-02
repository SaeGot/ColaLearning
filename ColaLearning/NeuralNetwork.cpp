#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(vector<Layer> _layers)
{
	layers = _layers;
	InitWeights();
}

NeuralNetwork::NeuralNetwork(vector<Layer> _layers, vector<Weight> _weights)
{
	layers = _layers;
	weights = _weights;
}

vector<double> NeuralNetwork::Predict(Layer layer)
{
	FeedForward(layer);

	return layers[layers.size() - 1].GetNodeValue();
}

void NeuralNetwork::Learn(vector<Layer> input_Layers, vector<Layer> output_Layer)
{
	for (const Layer& layer : input_Layers)
	{
		FeedForward(layer);
		BackPropagation();
	}
}

void NeuralNetwork::InitWeights()
{
	int prev_node_count = 0;
	int next_node_count;
	for (const Layer &layer : layers)
	{
		next_node_count = layer.GetNodeCount();
		Weight weight(prev_node_count, next_node_count);
		weights.push_back(weight);
		// 이전 층 노드 수
		prev_node_count = layer.GetNodeCount() + 1;
	}
}

double NeuralNetwork::Sum(const Layer& layer, const Weight& weight, int j)
{
	double output_sum = 0.0;
	for (int i = 0; i < layer.GetNodeCount(); i++)
	{
		output_sum += layer.GetNodeValue(i) * weight.GetWeight(i, j);
	}
	if (layer.CheckBias())
	{
		output_sum += weight.GetWeight(layer.GetNodeCount(), j);
	}

	return output_sum;
}

void NeuralNetwork::FeedForward(Layer layer)
{
	vector<double> predict_values;

	layers[0] = layer;
	for (size_t index = 1; index < layers.size(); index++)
	{
		layers[index].InitNodeValue();
		for (int j = 0; j < layers[index].GetNodeCount(); j++)
		{
			double value = Sum(layers[index - 1], weights[index - 1], j);
			value = layers[index].Activate(value);
			layers[index].SetNodeValue(j, value);
		}
	}
}

void NeuralNetwork::BackPropagation()
{
	// ToDo
	// (y - t) * next(da * w) * da * x
}
