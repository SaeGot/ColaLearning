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
			weightValues[TensorConnection( Tensor::Bias(), Tensor(j) )] = weight_Values[previous_index][j];
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
		weightValues[TensorConnection(Tensor::Bias(), Tensor(j))] = Initialize(init_Weight, input_NodeCountWithBias, output_NodeCount, initial_Limit);
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
	case Layer::LayerType::Pooling:
		ConnectPoolingLayer(previous_Layer, next_Layer, init_Weight, initial_Limit);
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

vector<Tensor> Weight::GetIConnectedWeightTensorWithoutBias(Tensor i) const
{
	vector<Tensor> i_weight_tensor;
	for (const pair<TensorConnection, double>& weight_values : weightValues)
	{
		if (weight_values.first.GetPrevious() == i && weight_values.first.GetPrevious() != Tensor::Bias())
		{
			i_weight_tensor.push_back(weight_values.first.GetNext());
		}
	}

	return i_weight_tensor;
}

vector<Tensor> Weight::GetJConnectedWeightTensorWithoutBias(Tensor j) const
{
	vector<Tensor> j_weight_tensor;
	for (const pair<TensorConnection, double>& weight_values : weightValues)
	{
		if (weight_values.first.GetNext() == j)
		{
			if (weight_values.first.GetPrevious() != Tensor::Bias())
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
			weightValues[TensorConnection(Tensor::Bias(), Tensor(j))]
				= Initialize(init_Weight, input_node_count_with_bias, output_node_count, initial_Limit);
		}
	}
}

void Weight::ConnectConvolutionLayer(Layer* previous_Layer, Layer* next_Layer, InitWeight init_Weight, double initial_Limit)
{
	// ConvolutionLayer의 가중치는 다음층 기준으로(이전층 Tensor의 x, y, bias 는 0)
	ConvolutionLayer* next_ConvolutionLayer = (ConvolutionLayer*)next_Layer;
	int prev_channel_size = previous_Layer->GetLayerSize()[2];
	int next_channel_size = next_ConvolutionLayer->GetLayerSize()[2];
	int input_node_count_with_bias = previous_Layer->GetNodeCount() + previous_Layer->CheckBias();
	int output_node_count = next_ConvolutionLayer->GetNodeCount();
	for (int prev_channel = 0; prev_channel < prev_channel_size; prev_channel++)
	{
		Tensor prev_tensor = Tensor(0, 0, prev_channel);
		for (int next_channel = 0; next_channel < next_channel_size; next_channel++)
		{
			for (Tensor j : next_ConvolutionLayer->GetFilter().GetTensors())
			{
				Tensor next_tensor = j + Tensor(0, 0, next_channel);
				weightValues.emplace(TensorConnection(prev_tensor, next_tensor),
					Initialize(init_Weight, input_node_count_with_bias, output_node_count, initial_Limit));
			}
			// 편향
			if (previous_Layer->CheckBias())
			{
				weightValues.emplace(TensorConnection(prev_tensor, Tensor(0, 0, next_channel, true)),
					Initialize(init_Weight, input_node_count_with_bias, output_node_count, initial_Limit));
			}
		}
	}

	Tensor filter_size_tensor = next_ConvolutionLayer->GetFilter();
	Tensor prev_layer_size = previous_Layer->GetLayerSizeTensor();
	vector<int> padding = next_ConvolutionLayer->GetPadding().GetXYChannel();
	vector<int> stride = next_ConvolutionLayer->GetStride().GetXYChannel();
	int x_stride = -padding[0];
	int y_stride = -padding[1];
	for (Tensor j : next_ConvolutionLayer->GetTensorWithoutBias())
	{
		vector<Tensor> all_prev_tensor;
		vector<Tensor> all_filter_tensor;
		for (int channel = 0; channel < prev_channel_size; channel++)
		{
			for (Tensor f : filter_size_tensor.GetTensors())
			{
				Tensor i = Tensor(x_stride, y_stride, channel);
				Tensor prev_tensor = i + f;
				if (!prev_tensor.CheckNegative() && prev_tensor.CheckOver(prev_layer_size))
				{
					all_prev_tensor.push_back(prev_tensor);
					all_filter_tensor.push_back(f);
				}
			}
			/*
			if (previous_Layer->CheckBias())
			{
				all_prev_tensor.push_back(Tensor(0, 0, channel, true));
			}
			*/
		}
		next_ConvolutionLayer->AddWeightConnection(Tensor(j), all_prev_tensor, all_filter_tensor);
		if (j.GetXYChannel()[2] == next_channel_size - 1)
		{
			x_stride += stride[0];
			if (x_stride + filter_size_tensor.GetXYChannel()[0] > next_ConvolutionLayer->GetLayerSize()[0])
			{
				x_stride = -padding[0];
				y_stride += stride[1];
			}
		}
	}
}

void Weight::ConnectPoolingLayer(Layer* previous_Layer, Layer* next_Layer, InitWeight init_Weight, double initial_Limit)
{
	PoolingLayer* next_ConvolutionLayer = (PoolingLayer*)next_Layer;
	int prev_node_count_x = previous_Layer->GetLayerSize()[0];
	int prev_node_count_y = previous_Layer->GetLayerSize()[1];
	int prev_channel = previous_Layer->GetLayerSize()[2];
	Tensor stride = next_ConvolutionLayer->GetStride();
	int stride_x = stride.GetXYChannel()[0];
	int stride_y = stride.GetXYChannel()[1];
	Tensor pool = next_ConvolutionLayer->GetPoolSize();
	int pool_x_size = pool.GetXYChannel()[0];
	int pool_y_size = pool.GetXYChannel()[1];
	int next_x = 0;
	int next_y = 0;
	for (int y = 0; y < prev_node_count_y; y++)
	{
		if (y % stride_y == 0)
		{
			for (int x = 0; x < prev_node_count_x; x++)
			{
				if (x % stride_x == 0)
				{
					Tensor next_tensor = Tensor(next_x, next_y);
					vector<Tensor> all_prev_tensor;
					for (int pool_x = 0; pool_x < pool_x_size; pool_x++)
					{
						for (int pool_y = 0; pool_y < pool_y_size; pool_y++)
						{
							for (int channel = 0; channel < prev_channel; channel++)
							{
								Tensor prev_tensor = Tensor(x + pool_x, y + pool_y, channel);
								weightValues.emplace(TensorConnection(prev_tensor, next_tensor), 0);
								all_prev_tensor.push_back(prev_tensor);
							}
						}
					}
					next_ConvolutionLayer->AddWeightConnection(next_tensor, all_prev_tensor);
					next_x++;
				}
			}

			next_x = 0;
			next_y++;
		}
	}
}
