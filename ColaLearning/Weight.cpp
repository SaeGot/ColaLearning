#include "Weight.h"


Weight::Weight(vector<double> weight_Values)
{
	weightValues = weight_Values;
}

Weight::Weight(const Weight &weight)
{
	weightValues = weight.weightValues;
}

double Weight::GetWeight(int n)
{
	if (n < 0)
	{
		printf("Error : %d번째 인자를 선택하였습니다.\n", n);
		return 0;
	}
	return weightValues[n];
}