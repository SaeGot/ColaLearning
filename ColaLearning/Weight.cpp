#include "Weight.h"
#include <random>


Weight::Weight(vector<vector<double>> weight_Values)
{
	weightValues = weight_Values;
}

Weight::Weight(int input_NodeCountWithBias, int output_NodeCount, InitWeight init_Weight, double initial_Limit)
{
	for (int i = 0; i < input_NodeCountWithBias; i++)
	{
		vector<double> next_node(output_NodeCount);
		for (double& init_value : next_node)
		{
			init_value = Initialize(init_Weight, input_NodeCountWithBias, output_NodeCount, initial_Limit);
		}
		weightValues.push_back(next_node);
	}
}

Weight::Weight(const Weight &weight)
{
	weightValues = weight.weightValues;
}

Weight::~Weight()
{
	weightValues.clear();
}

double Weight::GetWeight(int i, int j) const
{
	if (i < 0 || j < 0)
	{
		printf("Error : %d, %d번째 인자를 선택하였습니다.\n", i, j);
		return 0;
	}
	return weightValues[i][j];
}

void Weight::UpdateWeight(int i, int j, double value)
{
	weightValues[i][j] = value;
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
