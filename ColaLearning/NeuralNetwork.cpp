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

vector<double> NeuralNetwork::Predict(Layer input_Layer)
{
	FeedForward(input_Layer);

	return layers[layers.size() - 1].GetNodeValue();
}

vector<double> NeuralNetwork::GetError(Layer input_Layer, vector<double> target_Values)
{
	vector<double> errors;
	errors.resize(layers[layers.size() - 1].GetNodeCount());
	if (errors.size() == target_Values.size())
	{
		FeedForward(input_Layer);
		for (int index = 0; index < input_Layer.GetNodeCount(); index++)
		{
			layers[layers.size() - 1].GetNodeValue(index) - target_Values[index];
		}
	}

	return errors;
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
	for (int index = 1; index < layers.size(); index++)
	{
		layers[index].InitNodeValue();
		for (int j = 0; j < layers[index].GetNodeCount(); j++)
		{
			int prev_index = index - 1;
			double value = Sum(layers[prev_index], weights[prev_index], j);
			value = layers[index].Activate(value);
			layers[index].SetNodeValue(j, value);
		}
	}
}

void NeuralNetwork::BackPropagation()
{
	// ToDo
	// (y - t) * next(da * w) * da * x
	for (int index = layers.size() - 1; index >= 0; index--)
	{
		for (int j = 0; j < layers[index].GetNodeCount(); j++)
		{
			int prev_index = index - 1;
			for (int i = 0; i < layers[prev_index].GetNodeCount(); i++)
			{
				// ToDo
			}
		}
		if (layers[index].CheckBias())
		{
			// ToDo
		}
	}
}
