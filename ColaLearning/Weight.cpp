#include "Weight.h"


Weight::Weight(vector<vector<double>> weight_Values)
{
	weightValues = weight_Values;
}

Weight::Weight(int prev_node_count, int next_node_count)
{
	vector<double> next_node(next_node_count);
	for (int n = 0; n < next_node_count; n++)
	{
		weightValues.push_back(next_node);
	}
}

Weight::Weight(const Weight &weight)
{
	weightValues = weight.weightValues;
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