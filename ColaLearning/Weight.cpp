#include "Weight.h"


Weight::Weight(vector<vector<double>> weight_Values)
{
	weightValues = weight_Values;
}

Weight::Weight(int prev_NodeCountWithBias, int next_NodeCount)
{
	vector<double> next_node(next_NodeCount);
	for (int n = 0; n < prev_NodeCountWithBias; n++)
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
	weightValues[i][j] -= value;
}
