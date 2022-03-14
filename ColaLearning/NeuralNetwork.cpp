#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(const Layer* _layers, int layer_Count, double weight_InitialLimit)
{
	layerCount = layer_Count;
	layers = new Layer[layerCount];
	for (int n = 0; n < layerCount; n++)
	{
		layers[n] = _layers[n];
	}
	InitWeights(weight_InitialLimit);
	minMaxSet = false;
}

NeuralNetwork::NeuralNetwork(const Layer* _layers, int layer_Count, const Weight* _weights)
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
	minMaxSet = false;
}

NeuralNetwork::~NeuralNetwork()
{
	delete[] weights;
}

vector<double> NeuralNetwork::Predict(const Layer& input_Layer)
{
	Layer output;
	if (minMaxSet)
	{
		Layer normaized_input = GetNormalized(input_Layer, inputNodeMinMax);
		FeedForward(normaized_input);
		output = GetDenormalized(layers[layerCount - 1], outputNodeMinMax);
	}
	else
	{
		FeedForward(input_Layer);
		output = layers[layerCount - 1];
	}

	return output.GetNodeValue();
}

void NeuralNetwork::Learn(vector<Layer> input_Layers, vector<Layer> target_Layers, Optimizer* optimizer, int repeat)
{
	SetMinMax(input_Layers, target_Layers);
	if (input_Layers.size() == target_Layers.size())
	{
		while (repeat>0)
		{
			for (int n = 0; n < input_Layers.size(); n++)
			{
				Layer normaized_input = GetNormalized(input_Layers[n], inputNodeMinMax);
				Layer normaized_target = GetNormalized(target_Layers[n], outputNodeMinMax);
				vector<double> errors = FeedForward(normaized_input, normaized_target);
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

void NeuralNetwork::InitWeights(double weight_InitialLimit)
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
			case Layer::ActivationFunction::Linear:
			case Layer::ActivationFunction::ReLU:
				init_weight = InitWeight::He;
				break;
			case Layer::ActivationFunction::Step:
				init_weight = InitWeight::Xavier;
				break;
			}
			Weight weight(prev_node_count_with_bias, next_node_count, init_weight, weight_InitialLimit);
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

double NeuralNetwork::ForwardSum(const Layer& layer, const Weight& weight, Tensor j)
{
	double output_sum = 0.0;
	for (const Tensor& tensor : layer.GetTensorWithoutBias())
	{
		output_sum += layer.GetNodeValue(tensor) * weight.GetWeight(tensor, j);
	}
	/*
	for (int i = 0; i < layer.GetNodeCount(); i++)
	{
		output_sum += layer.GetNodeValue(i) * weight.GetWeight(i, j);
	}
	*/
	if (layer.CheckBias())
	{
		output_sum += weight.GetWeight(Tensor::GetBias(), j);
	}

	return output_sum;
}

void NeuralNetwork::FeedForward(const Layer &input_Layer)
{
	layers[0] = input_Layer;
	for (int n = 1; n < layerCount; n++)
	{
		for (Tensor j : layers[n].GetTensorWithoutBias())
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
	// 출력층 직전의 층까지
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

void NeuralNetwork::UpdateWeight(Weight& weight, const Layer& prev_Layer, Tensor i,
	Layer& next_Layer, Tensor j, Optimizer* optimizer)
{
	// backnode * x
	double backnode_value = next_Layer.GetBackNodeValue(j);
	double update_value = optimizer->GetUpdateValue(weight.GetWeight(i, j), backnode_value, prev_Layer.GetNodeValue(i));
	weight.UpdateWeight(i, j, update_value);
}

void NeuralNetwork::UpdateBiasWeight(Weight& weight, Layer& next_Layer, Tensor j, Optimizer* optimizer)
{
	// backnode * 1
	double update_value = optimizer->GetUpdateValue(weight.GetWeight(Tensor::GetBias(), j), next_Layer.GetBackNodeValue(j), 1);
	weight.UpdateWeight(Tensor::GetBias(), j, update_value);
}

void NeuralNetwork::BackPropagation(vector<double> errors, Optimizer* optimizer)
{
	// 출력층과 출력 이전 층 사이의 가중치 업데이트
	for (int j = 0; j < layers[layerCount - 1].GetNodeCount(); j++)
	{
		double d_activate = layers[layerCount - 1].Deactivate(layers[layerCount - 1].GetNodeValue(j));
		layers[layerCount - 1].SetBackNodeValue(j, errors[j] * d_activate);
		for (int i = 0; i < layers[layerCount - 2].GetNodeCount(); i++)
		{
			UpdateWeight(weights[layerCount - 2], layers[layerCount - 2], i,
				layers[layerCount - 1], j, optimizer);
		}
		if (layers[layerCount - 2].CheckBias())
		{
			UpdateBiasWeight(weights[layerCount - 2], layers[layerCount - 1], j, optimizer);
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
			double d_activate = layers[n].Deactivate(layers[n].GetNodeValue(j));
			layers[n].SetBackNodeValue(j, sum * d_activate);
			
			for (int i = 0; i < layers[prev_n].GetNodeCount(); i++)
			{
				UpdateWeight(weights[prev_n], layers[prev_n], i,
					layers[n], j, optimizer);
			}
			if (layers[n].CheckBias())
			{
				UpdateBiasWeight(weights[prev_n], layers[n], j, optimizer);
			}
		}
	}
}

void NeuralNetwork::SetMinMax(vector<Layer> input_Layers, vector<Layer> target_Layers)
{
	if (!minMaxSet)
	{
		inputNodeMinMax.resize(input_Layers[0].GetNodeCount());
		outputNodeMinMax.resize(target_Layers[0].GetNodeCount());
		for (int n = 0; n < input_Layers.size(); n++)
		{
			for (int node_index = 0; node_index < input_Layers[n].GetNodeCount(); node_index++)
			{
				inputNodeMinMax[node_index].min = min(inputNodeMinMax[node_index].min, input_Layers[n].GetNodeValue(node_index));
				inputNodeMinMax[node_index].max = max(inputNodeMinMax[node_index].max, input_Layers[n].GetNodeValue(node_index));
			}
			for (int node_index = 0; node_index < target_Layers[n].GetNodeCount(); node_index++)
			{
				outputNodeMinMax[node_index].min = min(outputNodeMinMax[node_index].min, target_Layers[n].GetNodeValue(node_index));
				outputNodeMinMax[node_index].max = max(outputNodeMinMax[node_index].max, target_Layers[n].GetNodeValue(node_index));
			}
		}
		// 정규화 기준을 첫 학습에만 설정
		minMaxSet = true;
	}
}

Layer NeuralNetwork::GetNormalized(const Layer& layer, vector<MinMax> min_Max)
{
	Layer normalized_layer = layer;
	for (int n = 0; n < normalized_layer.GetNodeCount(); n++)
	{
		double normalized_value = (layer.GetNodeValue(n) - min_Max[n].min) / (min_Max[n].max - min_Max[n].min);
		normalized_layer.SetNodeValue(n, normalized_value);
	}

	return normalized_layer;
}

Layer NeuralNetwork::GetDenormalized(const Layer& layer, vector<MinMax> min_Max)
{
	Layer denormalized_layer = layer;
	for (int n = 0; n < denormalized_layer.GetNodeCount(); n++)
	{
		double denormalized_value = layer.GetNodeValue(n) * (min_Max[n].max - min_Max[n].min) + min_Max[n].min;
		denormalized_layer.SetNodeValue(n, denormalized_value);
	}

	return denormalized_layer;
}
