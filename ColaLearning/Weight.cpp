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
		break;
	case Layer::LayerType::Convolution:
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

map<TensorConnection, double> Weight::GetJWeightValues(Tensor j) const
{
	map<TensorConnection, double> j_weight_values;
	for (const pair<TensorConnection, double>& weight_values : weightValues)
	{
		if (weight_values.first.GetNext() == j)
		{
			j_weight_values.emplace(weight_values.first, weight_values.second);
		}
	}

	return j_weight_values;
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
