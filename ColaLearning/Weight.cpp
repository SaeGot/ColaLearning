#include "Weight.h"


Weight::Weight(vector<double> weight_Values)
{
	weightValues = weight_Values;
}

double Weight::GetWeight(int n)
{
	if (n < 0)
	{
		printf("Error : %d��° ���ڸ� �����Ͽ����ϴ�.\n", n);
		return 0;
	}
	return weightValues[n];
}