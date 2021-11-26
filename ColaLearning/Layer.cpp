#include "Layer.h"


Layer::Layer(vector<double> node_Values)
{
	nodeValues = node_Values;
}

double Layer::GetNodeValue(int n)
{
	if (n < 0)
	{
		printf("Error : %d번째 인자를 선택하였습니다.\n", n);
		return 0;
	}
	return nodeValues[n];
}

int Layer::GetNodeCount()
{
	return nodeValues.size();
}