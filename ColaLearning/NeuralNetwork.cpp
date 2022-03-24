#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(vector<Layer*> _layers, int layer_Count, double weight_InitialLimit)
{
	layerCount = layer_Count;
	layers = vector<Layer*>(layerCount);
	for (int n = 0; n < layerCount; n++)
	{
		if (_layers[n]->GetLayerType() == Layer::LayerType::Convolution || _layers[n]->GetLayerType() == Layer::LayerType::Pooling)
		{
			int prev_n = n - 1;
			_layers[n]->SetNodes(*_layers[prev_n]);
		}
		layers[n] = _layers[n];
	}
	InitWeights(weight_InitialLimit);
	minMaxSet = false;
}

NeuralNetwork::NeuralNetwork(vector<Layer*> _layers, int layer_Count, const Weight* _weights)
{
	layerCount = layer_Count;
	layers = vector<Layer*>(layerCount);
	for (int n = 0; n < layerCount; n++)
	{
		if (_layers[n]->GetLayerType() == Layer::LayerType::Convolution)
		{
			int prev_n = n - 1;
			layers[n]->SetNodes(*_layers[prev_n]);
		}
		layers[n] = _layers[n];
	}
	int weight_count = layerCount - 1;
	weights.resize(weight_count);
	for (int n = 0; n < weight_count; n++)
	{
		weights[n] = _weights[n];
	}
	minMaxSet = false;
}

NeuralNetwork::~NeuralNetwork()
{
	layers.clear();
	weights.clear();
}

map<Tensor, double> NeuralNetwork::Predict(const Layer& input_Layer)
{
	Layer output;
	int final_index = layerCount - 1;
	if (minMaxSet)
	{
		Layer normaized_input = GetNormalized(&input_Layer, inputNodeMinMax);
		FeedForward(normaized_input);
		output = GetDenormalized(layers[final_index], outputNodeMinMax);
	}
	else
	{
		FeedForward(input_Layer);
		output = *layers[final_index];
	}

	return output.GetNodeValue();
}

void NeuralNetwork::Learn(vector<Layer*> input_Layers, vector<Layer*> target_Layers, Optimizer* optimizer, ErrorType error_Type, int repeat)
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
				map<Tensor, double> errors = FeedForward(normaized_input, normaized_target);
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
	int weight_count = layerCount - 1;
	weights.resize(weight_count);
	//int prev_node_count_with_bias = 0;
	//int next_node_count;
	bool input = true;
	int n = 0;
	for (int n = 0; n < layerCount; n++)
	{
		//const Layer& layer = layers[n];
		//next_node_count = layer.GetNodeCount();
		if (input)
		{
			input = false;
		}
		else
		{
			//{}로 초기화
			InitWeight init_weight{};
			switch (layers[n]->GetActivationFunction())
			{
			case Layer::ActivationFunction::Linear:
			case Layer::ActivationFunction::ReLU:
				init_weight = InitWeight::He;
				break;
			case Layer::ActivationFunction::Step:
			case Layer::ActivationFunction::Tanh:
				init_weight = InitWeight::Xavier;
				break;
			}
			int prev_n = n - 1;
			Weight weight(layers[prev_n], layers[n], init_weight, weight_InitialLimit);
			weights[prev_n] = weight;
		}
		/*
		// 이전 층 노드 수
		if (layer.CheckBias())
		{
			prev_node_count_with_bias = layer.GetNodeCount() + 1;
		}
		else
		{
			prev_node_count_with_bias = layer.GetNodeCount();
		}
		*/
	}
}

double NeuralNetwork::ForwardSum(const Layer& prev_Layer, const Layer& next_Layer, Weight* weight, Tensor j)
{
	if (next_Layer.GetLayerType() == Layer::LayerType::FullyConnected)
	{
		return FullyConnectedForwardSum(prev_Layer, *weight, j);
	}
	else if (next_Layer.GetLayerType() == Layer::LayerType::Convolution)
	{
		return ConvolutionForwardSum(prev_Layer, next_Layer, *weight, j);
	}
	else if (next_Layer.GetLayerType() == Layer::LayerType::Pooling)
	{
		return PoolingForwardSum(prev_Layer, next_Layer, weight, j);
	}
}

double NeuralNetwork::FullyConnectedForwardSum(const Layer& prev_Layer, const Weight& weight, Tensor j)
{
	double output_sum = 0.0;
	for (const Tensor& tensor : weight.GetJConnectedWeightTensorWithoutBias(j))
	{
		output_sum += prev_Layer.GetNodeValue(tensor) * weight.GetWeight(tensor, j);
	}
	/*
	for (int i = 0; i < layer.GetNodeCount(); i++)
	{
		output_sum += layer.GetNodeValue(i) * weight.GetWeight(i, j);
	}
	*/
	if (prev_Layer.CheckBias())
	{
		output_sum += weight.GetWeight(Tensor::Bias(), j);
	}

	return output_sum;
}

double NeuralNetwork::ConvolutionForwardSum(const Layer& prev_Layer, const Layer& next_Layer, const Weight& weight, Tensor j)
{
	double output_sum = 0.0;

	ConvolutionLayer* next_ConvolutionLayer = (ConvolutionLayer*)&next_Layer;
	vector<Tensor> prev_tensor_list = next_ConvolutionLayer->GetPreviousConnectionWithoutBias(j);
	vector<Tensor> filter_tensor_list = next_ConvolutionLayer->GetFilterConnectionWithoutBias(j);
	for (int i = 0; i < prev_tensor_list.size(); i++)
	{
		Tensor prev_tensor = prev_tensor_list[i];
		Tensor filter_tensor = filter_tensor_list[i];
		int prev_channel = prev_tensor_list[i].GetXYChannel()[2];
		output_sum += prev_Layer.GetNodeValue(prev_tensor) * weight.GetWeight(Tensor(0, 0, prev_channel), filter_tensor);
	}
	if (prev_Layer.CheckBias())
	{
		int prev_channel_size = prev_Layer.GetLayerSize()[2];
		int next_channel = j.GetXYChannel()[2];
		for (int prev_channel = 0; prev_channel < prev_channel_size; prev_channel++)
		{
			output_sum += weight.GetWeight(Tensor(0, 0, prev_channel), Tensor(0, 0, next_channel, true));
		}
	}



	/*
	vector<int> padding = next_ConvolutionLayer->GetPadding().GetXYChannel();
	vector<int> stride = next_ConvolutionLayer->GetStride().GetXYChannel();
	int prev_channel_size = prev_Layer.GetLayerSizeTensor().GetXYChannel()[2];
	int next_channel_size = next_ConvolutionLayer->GetLayerSizeTensor().GetXYChannel()[2];
	Tensor filter_size_tensor = next_ConvolutionLayer->GetFilter();

	Tensor prev_layer_size = prev_Layer.GetLayerSizeTensor();
	int x_stride = -padding[0];
	int y_stride = -padding[1];

	for (int prev_channel = 0; prev_channel < prev_channel_size; prev_channel++)
	{
		for (int next_channel = 0; next_channel < next_channel_size; next_channel++)
		{
			for (Tensor f : filter_size_tensor.GetTensors())
			{
				Tensor i = Tensor(x_stride, y_stride, prev_channel);
				Tensor prev_tensor = i + f;
				if (!prev_tensor.CheckNegative() && prev_tensor.CheckOver(prev_layer_size))
				{
					output_sum += prev_Layer.GetNodeValue(prev_tensor) * weight.GetWeight(prev_tensor, f);
				}
			}
			if (prev_Layer.CheckBias())
			{
				output_sum += weight.GetWeight(Tensor(0, 0, prev_channel, true), Tensor(0, 0, next_channel));
			}
		}
	}
	*/
	return output_sum;
}

double NeuralNetwork::PoolingForwardSum(const Layer& prev_Layer, const Layer& next_Layer, Weight* weight, Tensor j)
{
	PoolingLayer* next_ConvolutionLayer = (PoolingLayer*)&next_Layer;
	vector<Tensor> prev_tensors = next_ConvolutionLayer->GetPreviousConnectionWithoutBias(j);
	double final_value = 0;
	bool first = true;
	Tensor max_index = 0;
	if (next_ConvolutionLayer->GetPoolingType() == PoolingLayer::PoolingType::Max)
	{
		for (const Tensor& prev_tensor : prev_tensors)
		{
			double value = prev_Layer.GetNodeValue(prev_tensor);
			if (first)
			{
				first = false;
				final_value = value;
				max_index = prev_tensor;
			}
			else if (value > final_value)
			{
				final_value = value;
				max_index = prev_tensor;
			}
			weight->UpdateWeight(prev_tensor, j, 0.0);
		}
		weight->UpdateWeight(max_index, j, 1.0);

		return final_value;
	}
}

void NeuralNetwork::FeedForward(const Layer& input_Layer)
{
	layers[0] = new Layer(input_Layer);
	for (int n = 1; n < layerCount; n++)
	{
		for (Tensor j : layers[n]->GetTensorWithoutBias())
		{
			int prev_n = n - 1;
			double value = ForwardSum(*layers[prev_n], *layers[n], &weights[prev_n], j);
			value = layers[n]->Activate(value);
			layers[n]->SetNodeValue(j, value);
		}
	}
	int n = layerCount - 1;
	if (layers[n]->GetActivationFunction() == Layer::ActivationFunction::Softmax)
	{
		double softmax_sum = 0;
		for (Tensor j : layers[n]->GetTensorWithoutBias())
		{
			int prev_n = n - 1;
			double value = ForwardSum(*layers[prev_n], *layers[n], &weights[prev_n], j);
			value = layers[n]->Activate(value);
			layers[n]->SetNodeValue(j, value);
			softmax_sum += value;
		}
		for (Tensor j : layers[n]->GetTensorWithoutBias())
		{
			double value = layers[n]->GetNodeValue(j) / softmax_sum;
			layers[n]->SetNodeValue(j, value);
		}
	}
	else
	{
		for (Tensor j : layers[n]->GetTensorWithoutBias())
		{
			int prev_n = n - 1;
			double value = ForwardSum(*layers[prev_n], *layers[n], &weights[prev_n], j);
			value = layers[n]->Activate(value);
			layers[n]->SetNodeValue(j, value);
		}
	}
}

map<Tensor, double> NeuralNetwork::FeedForward(const Layer& input_Layer, const Layer& target_Layer)
{	
	layers[0] = new Layer(input_Layer);
	// 출력층 직전의 층까지
	for (int n = 1; n < layerCount - 1; n++)
	{
		for (Tensor j : layers[n]->GetTensorWithoutBias())
		{
			int prev_n = n - 1;
			double value = ForwardSum(*layers[prev_n], *layers[n], &weights[prev_n], j);
			value = layers[n]->Activate(value);
			layers[n]->SetNodeValue(j, value);
		}
	}
	// 출력 층
	map<Tensor, double> errors;
	int n = static_cast<int>(layerCount - 1);
	if (target_Layer.GetActivationFunction() == Layer::ActivationFunction::Softmax)
	{
		double softmax_sum = 0;
		for (Tensor j : layers[n]->GetTensorWithoutBias())
		{
			int prev_n = n - 1;
			double value = ForwardSum(*layers[prev_n], *layers[n], &weights[prev_n], j);
			value = layers[n]->Activate(value);
			layers[n]->SetNodeValue(j, value);
			softmax_sum += value;
		}
		for (Tensor j : layers[n]->GetTensorWithoutBias())
		{
			double value = layers[n]->GetNodeValue(j) / softmax_sum;
			layers[n]->SetNodeValue(j, value);
			// 오차 계산
			errors.emplace(j, target_Layer.GetNodeValue(j) - layers[n]->GetNodeValue(j));
		}
	}
	else
	{
		for (Tensor j : layers[n]->GetTensorWithoutBias())
		{
			int prev_n = n - 1;
			double value = ForwardSum(*layers[prev_n], *layers[n], &weights[prev_n], j);
			value = layers[n]->Activate(value);
			layers[n]->SetNodeValue(j, value);
			// 오차 계산
			errors.emplace(j, target_Layer.GetNodeValue(j) - layers[n]->GetNodeValue(j));
		}
	}
	
	return errors;
}

double NeuralNetwork::BackwardSum(const Layer& prev_Layer, const Layer& next_Layer, const Weight& weight, Tensor i)
{
	if (next_Layer.GetLayerType() == Layer::LayerType::FullyConnected)
	{
		return FullyConnectedBackwardSum(next_Layer, weight, i);
	}
	else if (next_Layer.GetLayerType() == Layer::LayerType::Convolution)
	{
		return ConvolutionBackwardSum(prev_Layer, next_Layer, weight, i);
	}
	else if (next_Layer.GetLayerType() == Layer::LayerType::Pooling)
	{
		return PoolingBackwardSum(prev_Layer, next_Layer, weight, i);
	}
}

double NeuralNetwork::FullyConnectedBackwardSum(const Layer& next_Layer, const Weight& weight, Tensor i)
{
	double output_sum = 0.0;
	for (const Tensor& j : weight.GetIConnectedWeightTensorWithoutBias(i))
	{
		output_sum += next_Layer.GetBackNodeValue(j) * weight.GetWeight(i, j);
	}

	return output_sum;
}

double NeuralNetwork::ConvolutionBackwardSum(const Layer& prev_Layer, const Layer& next_Layer, const Weight& weight, Tensor i)
{
	double output_sum = 0.0;

	ConvolutionLayer* next_ConvolutionLayer = (ConvolutionLayer*)&next_Layer;
	vector<int> padding = next_ConvolutionLayer->GetPadding().GetXYChannel();
	vector<int> stride = next_ConvolutionLayer->GetStride().GetXYChannel();
	int prev_channel_size = prev_Layer.GetLayerSizeTensor().GetXYChannel()[2];
	int next_channel_size = next_ConvolutionLayer->GetLayerSizeTensor().GetXYChannel()[2];
	Tensor filter_size_tensor = next_ConvolutionLayer->GetFilter();
	vector<int> filter_size = filter_size_tensor.GetXYChannel();

	for (int next_channel = 0; next_channel < next_channel_size; next_channel++)
	{
		for (Tensor f : filter_size_tensor.GetTensors())
		{
			int prev_channel = i.GetXYChannel()[2];
			//Tensor prev_tensor = i + f;
			Tensor invert_f = f.GetInvertTensor(filter_size[0], filter_size[1]);
			output_sum += next_Layer.GetBackNodeValue(invert_f) * weight.GetWeight(Tensor(0, 0, prev_channel), f);
		}
	}

	return output_sum;
}

double NeuralNetwork::PoolingBackwardSum(const Layer& prev_Layer, const Layer& next_Layer, const Weight& weight, Tensor i)
{
	double output_sum = 0.0;
	
	PoolingLayer* next_ConvolutionLayer = (PoolingLayer*)&next_Layer;
	Tensor stride = next_ConvolutionLayer->GetStride();
	int prev_size_x = prev_Layer.GetLayerSize()[0];
	int prev_size_y = prev_Layer.GetLayerSize()[1];
	int stride_x = stride.GetXYChannel()[0];
	int stride_y = stride.GetXYChannel()[1];
	int x = i.GetXYChannel()[0];
	int y = i.GetXYChannel()[1];
	vector<Tensor> pools = next_ConvolutionLayer->GetPools();
	int next_x = 0;
	int next_y = 0;
	if (y % stride_y == 0 && x % stride_x == 0 && y < prev_size_y - stride_y && x < prev_size_x - stride_x)
	{
		for (const Tensor& p : pools)
		{
			Tensor j = Tensor(floorl(x / stride_x), floorl(y / stride_y));
			output_sum += next_Layer.GetBackNodeValue(j) * weight.GetWeight(i, j);
		}
	}

	return output_sum;
}

void NeuralNetwork::UpdateWeight(Weight& weight, const Layer& prev_Layer, Tensor i,
	Layer& next_Layer, Tensor j, Optimizer* optimizer)
{
	double backnode_value = next_Layer.GetBackNodeValue(j);
	double update_value = 0;
	// backnode * x
	update_value = optimizer->GetUpdateValue(weight.GetWeight(i, j), backnode_value, prev_Layer.GetNodeValue(i));	
	weight.UpdateWeight(i, j, update_value);
}

void NeuralNetwork::UpdateBiasWeight(Weight& weight, Layer& next_Layer, Tensor j, Optimizer* optimizer)
{
	double update_value = 0;
	// backnode * 1
	update_value = optimizer->GetUpdateValue(weight.GetWeight(Tensor::Bias(), j), next_Layer.GetBackNodeValue(j), 1);
	weight.UpdateWeight(Tensor::Bias(), j, update_value);
	weight.UpdateWeight(Tensor::Bias(), j, update_value);
}

void NeuralNetwork::UpdateConvolutionWeight(Weight& weight, const Layer& prev_Layer, const ConvolutionLayer& next_Layer, Tensor f, Optimizer* optimizer)
{
	double update_value = 0;
	for (int prev_channel = 0; prev_channel < prev_Layer.GetLayerSize()[2]; prev_channel++)
	{
		int count = 0;
		for (Tensor j : next_Layer.GetTensorWithoutBias())
		{
			vector<Tensor> prev_tensor_list = next_Layer.GetPreviousConnectionWithoutBias(j);
			vector<Tensor> filter_tensor_list = next_Layer.GetFilterConnectionWithoutBias(j);
			for (int n = 0; n < filter_tensor_list.size(); n++)
			{
				Tensor filter_tensor = filter_tensor_list[n];
				if (filter_tensor == f)
				{
					Tensor prev_tensor = prev_tensor_list[n];
					double backnode_value = next_Layer.GetBackNodeValue(j);
					update_value += optimizer->GetUpdateValue(weight.GetWeight(Tensor(0, 0, prev_channel), filter_tensor), backnode_value, prev_Layer.GetNodeValue(prev_tensor));
					count++;
				}
			}
		}
		weight.UpdateWeight(Tensor(0, 0, prev_channel), f, update_value / count);
	}
}

void NeuralNetwork::UpdateConvolutionBiasWeight(Weight& weight, const Layer& prev_Layer, const ConvolutionLayer& next_Layer, Optimizer* optimizer)
{
	double update_value = 0;
	// 확인 필요
	for (int prev_channel = 0; prev_channel < prev_Layer.GetLayerSize()[2]; prev_channel++)
	{
		int count = 0;
		int next_channel = 0;
		for (Tensor j : next_Layer.GetTensorWithoutBias())
		{
			next_channel = j.GetXYChannel()[2];

			Tensor filter_bias = Tensor(0, 0, next_channel, true);

			double backnode_value = next_Layer.GetBackNodeValue(j);
			update_value += optimizer->GetUpdateValue(weight.GetWeight(Tensor(0, 0, prev_channel), filter_bias), backnode_value, 1.0);
			count++;
		}
		Tensor f_bias = Tensor(0, 0, next_channel, true);
		weight.UpdateWeight(Tensor(0, 0, prev_channel), f_bias, update_value / count);
	}
}

void NeuralNetwork::BackPropagation(map<Tensor, double> errors, Optimizer* optimizer)
{
	// 출력층과 출력 이전 층 사이의 가중치 업데이트 (출력층은 Convolution이 될 수 없도록)
	int final_index = layerCount - 1;
	int final_prev = layerCount - 2;
	for (Tensor j : layers[final_index]->GetTensorWithoutBias())
	{
		double d_activate = layers[final_index]->Deactivate(layers[final_index]->GetNodeValue(j));
		// 오차 부호 조심
		layers[final_index]->SetBackNodeValue(j, -errors[j] * d_activate);
		for (Tensor i : weights[final_prev].GetJConnectedWeightTensorWithoutBias(j))
		{
			UpdateWeight(weights[final_prev], *layers[final_prev], i, *layers[final_index], j, optimizer);
		}
		if (layers[final_prev]->CheckBias())
		{
			UpdateBiasWeight(weights[final_prev], *layers[final_index], j, optimizer);
		}
	}
	// 그 외 가중치 업데이트
	for (int n = final_prev; n > 0; n--)
	{
		int next_n = n + 1;
		int prev_n = n - 1;
		if (layers[n]->GetLayerType() == Layer::LayerType::FullyConnected)
		{
			for (Tensor j : layers[n]->GetTensorWithoutBias())
			{
				double sum = BackwardSum(*layers[n], *layers[next_n], weights[n], j);
				double d_activate = layers[n]->Deactivate(layers[n]->GetNodeValue(j));
				layers[n]->SetBackNodeValue(j, sum * d_activate);

				for (Tensor i : weights[prev_n].GetJConnectedWeightTensorWithoutBias(j))
				{
					UpdateWeight(weights[prev_n], *layers[prev_n], i, *layers[n], j, optimizer);
				}
				if (layers[n]->CheckBias())
				{
					UpdateBiasWeight(weights[prev_n], *layers[n], j, optimizer);
				}
			}
		}
		else if (layers[n]->GetLayerType() == Layer::LayerType::Convolution)
		{
			ConvolutionLayer* next_ConvolutionLayer = (ConvolutionLayer*)layers[n];
			for (Tensor j : layers[n]->GetTensorWithoutBias())
			{
				double sum = BackwardSum(*layers[n], *layers[next_n], weights[n], j);
				double d_activate = layers[n]->Deactivate(layers[n]->GetNodeValue(j));
				layers[n]->SetBackNodeValue(j, sum * d_activate);
			}
			for (Tensor f : next_ConvolutionLayer->GetFilter().GetTensors())
			{
				UpdateConvolutionWeight(weights[prev_n], *layers[prev_n], *next_ConvolutionLayer, f, optimizer);
			}
			if (layers[n]->CheckBias())
			{
				UpdateConvolutionBiasWeight(weights[prev_n], *layers[prev_n], *next_ConvolutionLayer, optimizer);
			}
		}
		else if (layers[n]->GetLayerType() == Layer::LayerType::Pooling)
		{
			for (Tensor j : layers[n]->GetTensorWithoutBias())
			{
				double sum = BackwardSum(*layers[n], *layers[next_n], weights[n], j);
				double d_activate = layers[n]->Deactivate(layers[n]->GetNodeValue(j));
				layers[n]->SetBackNodeValue(j, sum * d_activate);
			}
		}
	}
}

void NeuralNetwork::SetMinMax(vector<Layer*> input_Layers, vector<Layer*> target_Layers)
{
	if (!minMaxSet)
	{
		for (int n = 0; n < input_Layers.size(); n++)
		{
			for (Tensor index : input_Layers[n]->GetTensorWithoutBias())
			{
				inputNodeMinMax[index].min = min(inputNodeMinMax[index].min, input_Layers[n]->GetNodeValue(index));
				inputNodeMinMax[index].max = max(inputNodeMinMax[index].max, input_Layers[n]->GetNodeValue(index));
			}
			for (Tensor index : target_Layers[n]->GetTensorWithoutBias())
			{
				outputNodeMinMax[index].min = min(outputNodeMinMax[index].min, target_Layers[n]->GetNodeValue(index));
				outputNodeMinMax[index].max = max(outputNodeMinMax[index].max, target_Layers[n]->GetNodeValue(index));
			}
		}
		// 정규화 기준을 첫 학습에만 설정
		minMaxSet = true;
	}
}

Layer NeuralNetwork::GetNormalized(const Layer* layer, map<Tensor, MinMax> min_Max)
{
	Layer normalized_layer = *layer;
	for (Tensor n : normalized_layer.GetTensorWithoutBias())
	{
		double normalized_value = 0;
		if (min_Max[n].max == min_Max[n].min)
		{
			normalized_value = 1;
		}
		else
		{
			normalized_value = (layer->GetNodeValue(n) - min_Max[n].min) / (min_Max[n].max - min_Max[n].min);
		}		
		normalized_layer.SetNodeValue(n, normalized_value);
	}

	return normalized_layer;
}

Layer NeuralNetwork::GetDenormalized(const Layer* layer, map<Tensor, MinMax> min_Max)
{
	Layer denormalized_layer = *layer;
	for (Tensor n : denormalized_layer.GetTensorWithoutBias())
	{
		double denormalized_value = layer->GetNodeValue(n) * (min_Max[n].max - min_Max[n].min) + min_Max[n].min;
		denormalized_layer.SetNodeValue(n, denormalized_value);
	}

	return denormalized_layer;
}
