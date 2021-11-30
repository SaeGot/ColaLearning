#include "Layer.h"


Layer::Layer(vector<double> node_Values, bool _bias)
{
	nodeValues = node_Values;
	bias = _bias;
}

Layer::Layer(int count, bool _bias)
{
	nodeValues.resize(count);
	bias = _bias;
}

Layer::Layer(const Layer& layer)
{
	nodeValues = layer.nodeValues;
	bias = layer.bias;
}

double Layer::GetNodeValue(int index) const
{
	if (index < 0)
	{
		printf("Error : %d번째 인자를 선택하였습니다.\n", index);
		return 0;
	}
	return nodeValues[index];
}

vector<double> Layer::GetNodeValue() const
{
	return nodeValues;
}

void Layer::SetNodeValue(int index, double value)
{
	nodeValues[index] = value;
}

void Layer::InitNodeValue()
{
	for (double value : nodeValues)
	{
		value = 0.0;
	}
}

int Layer::GetNodeCount() const
{
	return static_cast<int>(nodeValues.size());
}

bool Layer::CheckBias() const
{
	return bias;
}