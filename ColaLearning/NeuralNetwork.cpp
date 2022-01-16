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
			errors.push_back(layers[layers.size() - 1].GetNodeValue(index) - target_Layer.GetNodeValue(index));
		}
	}
	else
	{
		// Error
	}

	return errors;
}

void NeuralNetwork::Learn(vector<Layer> input_Layers, vector<Layer> target_Layers, Optimizer* optimizer, int repeat)
{
	if (input_Layers.size() == target_Layers.size())
	{
		while (repeat>0)
		{
			for (int index = 0; index < input_Layers.size(); index++)
			{
				vector<double> errors = FeedForward(input_Layers[index], target_Layers[index]);
				BackPropagation(errors, optimizer);
			}
			repeat--;
		}
	}
	else
	{
		// Error
	}
}

void NeuralNetwork::InitWeights()
{
	int prev_node_count_with_bias = 0;
	int next_node_count;
	bool input = true;
	for (const Layer &layer : layers)
	{
		next_node_count = layer.GetNodeCount();
		if (input)
		{
			input = false;
		}
		else
		{
			Weight weight(prev_node_count_with_bias, next_node_count);
			weights.push_back(weight);
		}
		// 이전 층 노드 수
		if (layer.CheckBias())
		{
			prev_node_count_with_bias = layer.GetNodeCount() + 1;
		}
		else
		{
			prev_node_count_with_bias = layer.GetNodeCount();
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
	int index = static_cast<int>(layers.size() - 1);
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

double NeuralNetwork::BackwardSum(const Layer& next_Layer, const Weight& weight, int i)
{
	double output_sum = 0.0;
	for (int j = 0; j < next_Layer.GetNodeCount(); j++)
	{
		output_sum += next_Layer.GetBackNodeValue(j) * weight.GetWeight(i, j);
	}

	return output_sum;
}

void NeuralNetwork::UpdateWeight(Weight& weight, const Layer& prev_Layer, int i,
	Layer& next_Layer, int j, Optimizer* optimizer)
{
	// backnode * x
	double backnode_value = next_Layer.GetBackNodeValue(j);
	double update_value = optimizer->GetUpdateValue(weight.GetWeight(i, j), backnode_value, prev_Layer.GetNodeValue(i));
	weight.UpdateWeight(i, j, update_value);
}

void NeuralNetwork::UpdateBiasWeight(Weight& weight, int i,
	Layer& next_Layer, int j, Optimizer* optimizer)
{
	// backnode * 1
	double update_value = optimizer->GetUpdateValue(weight.GetWeight(i, j), next_Layer.GetBackNodeValue(j), 1);
	weight.UpdateWeight(i, j, update_value);
}

void NeuralNetwork::BackPropagation(vector<double> errors, Optimizer* optimizer)
{
	// 출력층과 출력 이전 층 사이의 가중치 업데이트
	for (int j = 0; j < layers[layers.size() - 1].GetNodeCount(); j++)
	{
		double d_activate = layers[layers.size() - 1].DActivate(layers[layers.size() - 1].GetNodeValue(j));
		layers[layers.size() - 1].SetBackNodeValue(j, errors[j] * d_activate);
		for (int i = 0; i < layers[layers.size() - 2].GetNodeCount(); i++)
		{
			UpdateWeight(weights[layers.size() - 2], layers[layers.size() - 2], i,
				layers[layers.size() - 1], j, optimizer);
		}
		if (layers[layers.size() - 2].CheckBias())
		{
			UpdateBiasWeight(weights[layers.size() - 2], layers[layers.size() - 2].GetNodeCount(),
				layers[layers.size() - 1], j, optimizer);
		}
	}
	// 그 외 가중치 업데이트
	for (int index = static_cast<int>(layers.size() - 2); index > 0; index--)
	{
		int next_index = index + 1;
		int prev_index = index - 1;
		for (int j = 0; j < layers[index].GetNodeCount(); j++)
		{			
			double sum = BackwardSum(layers[next_index], weights[index], j);
			double d_activate = layers[index].DActivate(layers[index].GetNodeValue(j));
			layers[index].SetBackNodeValue(j, sum * d_activate);
			
			for (int i = 0; i < layers[prev_index].GetNodeCount(); i++)
			{
				UpdateWeight(weights[prev_index], layers[prev_index], i,
					layers[index], j, optimizer);
			}
			if (layers[index].CheckBias())
			{
				UpdateBiasWeight(weights[prev_index], layers[prev_index].GetNodeCount(),
					layers[index], j, optimizer);
			}
		}
	}
}
