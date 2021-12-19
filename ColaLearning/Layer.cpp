#include "Layer.h"


Layer::Layer(vector<double> node_Values, const ActivationFunction &activation_Function, bool _bias)
{
	nodeValues = node_Values;
	activationFunction = activation_Function;
	bias = _bias;
}

Layer::Layer(int count, const ActivationFunction &activation_Function, bool _bias)
{
	nodeValues.resize(count);
	activationFunction = activation_Function;
	bias = _bias;
}

Layer::Layer(const Layer& layer)
{
	nodeValues = layer.nodeValues;
	activationFunction = layer.activationFunction;
	bias = layer.bias;
}

Layer::~Layer()
{
	nodeValues.clear();
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

double Layer::Activate(double value)
{
	switch (activationFunction)
	{
	case ActivationFunction::ReLU:
		if (value >= 0) { return value; }
		else { return 0.0; }

	case ActivationFunction::Step:
		if (value >= 0) { return 1.0; }
		else { return 0.0; }
	}

	return value;
}
