#include "Weight.h"
#include <random>


Weight::Weight(vector<vector<double>> weight_Values)
{
	weightValues = weight_Values;
}

Weight::Weight(int prev_NodeCountWithBias, int next_NodeCount, InitWeight init_Weight)
{
	vector<double> next_node(next_NodeCount);
	for (double& init_value : next_node)
	{
		// ToDo 가중치 초기화
		init_value = Initialize(init_Weight);
	}
	for (int i = 0; i < prev_NodeCountWithBias; i++)
	{
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

double Weight::Initialize(InitWeight init_Weight)
{
	random_device rd;
	mt19937_64 gen(rd());
	uniform_real_distribution<double> random_value(-1, 1);

	switch (init_Weight)
	{
	case InitWeight::RamdomUniform:
		return random_value(gen);
	}
	return 1.0;
}
