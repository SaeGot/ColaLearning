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

vector<double> NeuralNetwork::GetError(Layer input_Layer, Layer target_Layer)
{
	vector<double> errors;
	errors.resize(layers[layers.size() - 1].GetNodeCount());
	if (errors.size() == target_Layer.GetNodeCount())
	{
		FeedForward(input_Layer);
		for (int index = 0; index < target_Layer.GetNodeCount(); index++)
		{
			layers[layers.size() - 1].GetNodeValue(index) - target_Layer.GetNodeValue(index);
		}
	}
	else
	{
		// Error
	}

	return errors;
}

void NeuralNetwork::Learn(vector<Layer> input_Layers, vector<Layer> target_Layers)
{
	if (input_Layers.size() == target_Layers.size())
	{
		for (int index = 0; index < input_Layers.size(); index++)
		{
			vector<double> errors = FeedForward(input_Layers[index], target_Layers[index]);
			BackPropagation(target_Layers[index], errors);
		}
	}
	else
	{
		// Error
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
		if (layer.CheckBias())
		{
			prev_node_count = layer.GetNodeCount() + 1;
		}
		else
		{
			prev_node_count = layer.GetNodeCount();
		}
	}
}

double NeuralNetwork::ForwardSum(const Layer& layer, const Weight& weight, int j)
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

void NeuralNetwork::FeedForward(const Layer &input_Layer)
{
	layers[0] = input_Layer;
	for (int index = 1; index < layers.size(); index++)
	{
		for (int j = 0; j < layers[index].GetNodeCount(); j++)
		{
			int prev_index = index - 1;
			double value = ForwardSum(layers[prev_index], weights[prev_index], j);
			value = layers[index].Activate(value);
			layers[index].SetNodeValue(j, value);
		}
	}
}

vector<double> NeuralNetwork::FeedForward(const Layer& input_Layer, const Layer& target_Layer)
{
	layers[0] = input_Layer;
	for (int index = 1; index < layers.size() - 1; index++)
	{
		for (int j = 0; j < layers[index].GetNodeCount(); j++)
		{
			int prev_index = index - 1;
			double value = ForwardSum(layers[prev_index], weights[prev_index], j);
			value = layers[index].Activate(value);
			layers[index].SetNodeValue(j, value);
		}
	}
	// 출력 층
	vector<double> errors;
	int index = layers.size() - 1;
	for (int j = 0; j < layers[index].GetNodeCount(); j++)
	{
		int prev_index = index - 1;
		double value = ForwardSum(layers[prev_index], weights[prev_index], j);
		value = layers[index].Activate(value);
		layers[index].SetNodeValue(j, value);
		// 오차 계산
		errors.push_back(layers[index].GetNodeValue(j) - target_Layer.GetNodeValue(j));
	}
	
	return errors;
}

void NeuralNetwork::BackPropagation(const Layer &target_Layer, vector<double> errors)
{
	// 출력층과 출력 이전 층의 가중치 업데이트
	for (int j = 0; j < layers[layers.size() - 1].GetNodeCount(); j++)
	{
		double d_activate = layers[layers.size() - 1].DActivate(layers[layers.size() - 1].GetNodeValue(j));
		layers[layers.size() - 1].SetBackNodeValue(j, d_activate);
		for (int i = 0; i < layers[layers.size() - 2].GetNodeCount(); i++)
		{
			UpdateWeight(weights[layers.size() - 1], layers[layers.size() - 2], i,
				layers[layers.size() - 1], j, errors[j]);
		}
		if (layers[layers.size() - 2].CheckBias())
		{
			UpdateBiasWeight(weights[layers.size() - 1], layers[layers.size() - 2].GetNodeCount(),
				layers[layers.size() - 1], j, errors[j]);
		}
	}

	for (int index = layers.size() - 2; index > 0; index--)
	{
		for (int j = 0; j < layers[index].GetNodeCount(); j++)
		{
			double d_activate = layers[index].DActivate(layers[index].GetNodeValue(j));
			layers[index].SetBackNodeValue(j, d_activate);
			int prev_index = index - 1;
			for (int i = 0; i < layers[prev_index].GetNodeCount(); i++)
			{
				// ToDo
				// update weight
			}
		}
		if (layers[index].CheckBias())
		{
			// ToDo
			// update weight
		}
	}
}

void NeuralNetwork::UpdateWeight(Weight &weight, const Layer &prev_Layer, int i,
	Layer &next_Layer, int j, double error)
{
	// (y - t) * backnode * x
	double backnode_value = next_Layer.GetBackNodeValue(j);
	double update_value = error * backnode_value * prev_Layer.GetNodeValue(i);
	weight.UpdateWeight(i, j, update_value);
}

void NeuralNetwork::UpdateBiasWeight(Weight& weight, int i,
	Layer& next_Layer, int j, double error)
{
	// (y - t) * backnode
	double backnode_value = next_Layer.GetBackNodeValue(j);
	double update_value = error * backnode_value;
	weight.UpdateWeight(i, j, update_value);
}
