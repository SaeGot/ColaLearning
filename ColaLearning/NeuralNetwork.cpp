#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(const Layer* _layers, int layer_Count)
{
	layerCount = layer_Count;
	layers = new Layer[layerCount];
	for (int n = 0; n < layerCount; n++)
	{
		layers[n] = _layers[n];
	}
	InitWeights();
}

NeuralNetwork::NeuralNetwork(const Layer* _layers, const Weight* _weights, int layer_Count)
{
	layerCount = layer_Count;
	layers = new Layer[layerCount];
	for (int n = 0; n < layerCount; n++)
	{
		layers[n] = _layers[n];
	}
	int weight_count = layerCount - 1;
	weights = new Weight[weight_count];
	for (int n = 0; n < layerCount - 1; n++)
	{
		weights[n] = _weights[n];
	}
}

NeuralNetwork::~NeuralNetwork()
{
	delete[] weights;
}

vector<double> NeuralNetwork::Predict(Layer input_Layer)
{
	FeedForward(input_Layer);

	return layers[layerCount - 1].GetNodeValue();
}

vector<double> NeuralNetwork::GetError(Layer input_Layer, Layer target_Layer)
{
	vector<double> errors;
	errors.resize(layers[layerCount - 1].GetNodeCount());
	if (errors.size() == target_Layer.GetNodeCount())
	{
		FeedForward(input_Layer);
		for (int n = 0; n < target_Layer.GetNodeCount(); n++)
		{
			errors.push_back(layers[layerCount - 1].GetNodeValue(n) - target_Layer.GetNodeValue(n));
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
			for (int n = 0; n < input_Layers.size(); n++)
			{
				vector<double> errors = FeedForward(input_Layers[n], target_Layers[n]);
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
	weights = new Weight[layerCount];
	int prev_node_count_with_bias = 0;
	int next_node_count;
	bool input = true;
	int n = 0;
	for (int n = 0; n < layerCount; n++)
	{
		const Layer& layer = layers[n];
		next_node_count = layer.GetNodeCount();
		if (input)
		{
			input = false;
		}
		else
		{
			InitWeight init_weight{};
			switch (layer.GetActivationFunction())
			{
			case ActivationFunction::Linear:
			case ActivationFunction::ReLU:
				init_weight = InitWeight::He;
				break;
			case ActivationFunction::Step:
				init_weight = InitWeight::Xavier;
				break;
			}
			Weight weight(prev_node_count_with_bias, next_node_count, init_weight);
			weights[n - 1] = weight;
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
	for (int n = 1; n < layerCount; n++)
	{
		for (int j = 0; j < layers[n].GetNodeCount(); j++)
		{
			int prev_n = n - 1;
			double value = ForwardSum(layers[prev_n], weights[prev_n], j);
			value = layers[n].Activate(value);
			layers[n].SetNodeValue(j, value);
		}
	}
}

vector<double> NeuralNetwork::FeedForward(const Layer& input_Layer, const Layer& target_Layer)
{
	layers[0] = input_Layer;
	for (int n = 1; n < layerCount - 1; n++)
	{
		for (int j = 0; j < layers[n].GetNodeCount(); j++)
		{
			int prev_n = n - 1;
			double value = ForwardSum(layers[prev_n], weights[prev_n], j);
			value = layers[n].Activate(value);
			layers[n].SetNodeValue(j, value);
		}
	}
	// 출력 층
	vector<double> errors;
	int n = static_cast<int>(layerCount - 1);
	for (int j = 0; j < layers[n].GetNodeCount(); j++)
	{
		int prev_n = n - 1;
		double value = ForwardSum(layers[prev_n], weights[prev_n], j);
		value = layers[n].Activate(value);
		layers[n].SetNodeValue(j, value);
		// 오차 계산
		errors.push_back(layers[n].GetNodeValue(j) - target_Layer.GetNodeValue(j));
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
	for (int j = 0; j < layers[layerCount - 1].GetNodeCount(); j++)
	{
		double d_activate = layers[layerCount - 1].DActivate(layers[layerCount - 1].GetNodeValue(j));
		layers[layerCount - 1].SetBackNodeValue(j, errors[j] * d_activate);
		for (int i = 0; i < layers[layerCount - 2].GetNodeCount(); i++)
		{
			UpdateWeight(weights[layerCount - 2], layers[layerCount - 2], i,
				layers[layerCount - 1], j, optimizer);
		}
		if (layers[layerCount - 2].CheckBias())
		{
			UpdateBiasWeight(weights[layerCount - 2], layers[layerCount - 2].GetNodeCount(),
				layers[layerCount - 1], j, optimizer);
		}
	}
	// 그 외 가중치 업데이트
	for (int n = static_cast<int>(layerCount - 2); n > 0; n--)
	{
		int next_n = n + 1;
		int prev_n = n - 1;
		for (int j = 0; j < layers[n].GetNodeCount(); j++)
		{			
			double sum = BackwardSum(layers[next_n], weights[n], j);
			double d_activate = layers[n].DActivate(layers[n].GetNodeValue(j));
			layers[n].SetBackNodeValue(j, sum * d_activate);
			
			for (int i = 0; i < layers[prev_n].GetNodeCount(); i++)
			{
				UpdateWeight(weights[prev_n], layers[prev_n], i,
					layers[n], j, optimizer);
			}
			if (layers[n].CheckBias())
			{
				UpdateBiasWeight(weights[prev_n], layers[prev_n].GetNodeCount(),
					layers[n], j, optimizer);
			}
		}
	}
}
