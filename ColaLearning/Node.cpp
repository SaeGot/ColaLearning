#include "Node.h"


Node::Node(vector<double> node_Values)
{
	nodeValues = node_Values;
}

double Node::GetNodeValue(int n)
{
	if (n < 0)
	{
		printf("Error : %d��° ���ڸ� �����Ͽ����ϴ�.\n", n);
		return 0;
	}
	return nodeValues[n];
}