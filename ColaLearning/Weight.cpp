#include "Weight.h"
#include <random>


Weight::Weight(vector<vector<double>> weight_Values, bool previous_Bias)
{
	Initialize();
	previousBias = previous_Bias;
	int previous_count = static_cast<int>( weight_Values.size() );
	for (int i = 0; i < previous_count - 1; i++)
	{
		for (int j = 0; j < weight_Values[i].size(); j++)
		{
			weightValues[TensorConnection( Tensor(i), Tensor(j) )] = weight_Values[i][j];
		}
	}
	// 편향 or 마지막
	int previous_index = previous_count - 1;
	for (int j = 0; j < weight_Values[previous_index].size(); j++)
	{
		if (previousBias)
		{
			weightValues[TensorConnection( Tensor::GetBias(), Tensor(j) )] = weight_Values[previous_index][j];
		}
		else
		{
			weightValues[TensorConnection( Tensor(previous_count - 1), Tensor(j) )] = weight_Values[previous_index][j];
		}
	}
}

Weight::Weight(int input_NodeCountWithBias, int output_NodeCount, InitWeight init_Weight, double initial_Limit)
{
	Initialize();
	previousBias = true;
	for (int i = 0; i < input_NodeCountWithBias - 1; i++)
	{
		for (int j = 0; j < output_NodeCount; j++)
		{
			weightValues[TensorConnection( Tensor(i), Tensor(j) )] = Initialize(init_Weight, input_NodeCountWithBias, output_NodeCount, initial_Limit);
		}
	}
	// 편향
	int previous_index = input_NodeCountWithBias - 1;
	for (int j = 0; j < output_NodeCount; j++)
	{
		weightValues[TensorConnection(Tensor::GetBias(), Tensor(j))] = Initialize(init_Weight, input_NodeCountWithBias, output_NodeCount, initial_Limit);
	}
}

Weight::Weight(Layer* previous_Layer, Layer* next_Layer, InitWeight init_Weight, double initial_Limit)
{
	previousBias = previous_Layer->CheckBias();
	switch (next_Layer->GetLayerType())
	{
	case Layer::LayerType::FullyConnected:
		ConnectFullyConnectedLayer(previous_Layer, next_Layer, init_Weight, initial_Limit);
		break;
	case Layer::LayerType::Convolution:
		ConnectConvolutionLayer(previous_Layer, next_Layer, init_Weight, initial_Limit);
		break;
	}
}

Weight::Weight(const Weight &weight)
{
	weightValues = weight.weightValues;
	previousBias = weight.previousBias;
}

Weight::Weight()
{
	Initialize();
}

Weight::~Weight()
{
	Initialize();
}

double Weight::GetWeight(Tensor i, Tensor j) const
{
	/*
	if (i < 0 || j < 0)
	{
		printf("Error : %d, %d번째 인자를 선택하였습니다.\n", i, j);
		return 0;
	}
	*/
	return weightValues.at( TensorConnection(i, j) );
}

vector<Tensor> Weight::GetJWeightTensorWithoutBias(Tensor j) const
{
	vector<Tensor> j_weight_tensor;
	for (const pair<TensorConnection, double>& weight_values : weightValues)
	{
		if (weight_values.first.GetNext() == j)
		{
			if (weight_values.first.GetPrevious() != Tensor::GetBias())
			{
				j_weight_tensor.push_back(weight_values.first.GetPrevious());
			}
		}
	}

	return j_weight_tensor;
}

void Weight::UpdateWeight(Tensor i, Tensor j, double value)
{
	weightValues[TensorConnection(i, j)] = value;
}

double Weight::Initialize(InitWeight init_Weight, int input_NodeCountWithBias, int output_NodeCount, double limit)
{
	random_device rd;
	mt19937_64 gen(rd());
	uniform_real_distribution<double> random_value(-limit, limit);

	switch (init_Weight)
	{
	case InitWeight::RamdomUniform:
		return random_value(gen);
	case InitWeight::He:
		return random_value(gen) * sqrt(2.0 / input_NodeCountWithBias);
	case InitWeight::Xavier:
		int n = input_NodeCountWithBias + output_NodeCount;
		return random_value(gen) * sqrt(6.0 / n);
	}
	return 1.0;
}

void Weight::Initialize()
{
	weightValues.clear();
	previousBias = false;
}

void Weight::ConnectFullyConnectedLayer(Layer* previous_Layer, Layer* next_Layer, InitWeight init_Weight, double initial_Limit)
{
	for (Tensor i : previous_Layer->GetTensorWithoutBias())
	{
		for (Tensor j : next_Layer->GetTensorWithoutBias())
		{
			int input_node_count_with_bias = previous_Layer->GetNodeCount() + previous_Layer->CheckBias();
			int output_node_count = next_Layer->GetNodeCount();
			weightValues[TensorConnection(Tensor(i), Tensor(j))]
				= Initialize(init_Weight, input_node_count_with_bias, output_node_count, initial_Limit);
		}
	}
	// 편향
	if (previous_Layer->CheckBias())
	{
		for (Tensor j : next_Layer->GetTensorWithoutBias())
		{
			int input_node_count_with_bias = previous_Layer->GetNodeCount() + previous_Layer->CheckBias();
			int output_node_count = next_Layer->GetNodeCount();
			weightValues[TensorConnection(Tensor::GetBias(), Tensor(j))]
				= Initialize(init_Weight, input_node_count_with_bias, output_node_count, initial_Limit);
		}
	}
}

void Weight::ConnectConvolutionLayer(Layer* previous_Layer, Layer* next_Layer, InitWeight init_Weight, double initial_Limit)
{
	ConvolutionLayer* next_ConvolutionLayer = (ConvolutionLayer*)next_Layer;
	Tensor filter_size_tensor = next_ConvolutionLayer->GetFilter();
	Tensor prev_layer_size = previous_Layer->GetLayerSizeTensor();
	int input_node_count_with_bias = previous_Layer->GetNodeCount() + previous_Layer->CheckBias();
	int output_node_count = next_ConvolutionLayer->GetNodeCount();
	vector<int> padding = next_ConvolutionLayer->GetPadding().GetXYChannel();
	int x_stride = -padding[0];
	int y_stride = -padding[1];
	vector<int> stride = next_ConvolutionLayer->GetStride().GetXYChannel();
	for (Tensor j : next_ConvolutionLayer->GetTensorWithoutBias())
	{
		for (Tensor f : filter_size_tensor.GetTensors())
		{
			Tensor i = Tensor(x_stride, y_stride);
			Tensor prev_tensor = i + f;
			if (!prev_tensor.CheckNegative() && prev_tensor.CheckOver(prev_layer_size))
			{
				weightValues[TensorConnection(Tensor(prev_tensor), Tensor(j))]
					= Initialize(init_Weight, input_node_count_with_bias, output_node_count, initial_Limit);
			}
		}
		x_stride += stride[0];
		if (x_stride + filter_size_tensor.GetXYChannel()[0] > next_ConvolutionLayer->GetLayerSize()[0])
		{
			x_stride = -padding[0];
			y_stride += stride[1];
		}
		if (y_stride + filter_size_tensor.GetXYChannel()[1] > next_ConvolutionLayer->GetLayerSize()[1])
		{
			break;
		}
	}
	// 편향
	if (previous_Layer->CheckBias())
	{
		for (Tensor j : next_ConvolutionLayer->GetTensorWithoutBias())
		{
			weightValues[TensorConnection(Tensor::GetBias(), Tensor(j))]
				= Initialize(init_Weight, input_node_count_with_bias, output_node_count, initial_Limit);
		}
	}
}
