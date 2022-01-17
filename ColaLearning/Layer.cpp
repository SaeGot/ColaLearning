#include "Layer.h"


Layer::Layer(vector<double> node_Values, const ActivationFunction &activation_Function, bool _bias)
{
	nodeValues = node_Values;
	backNodeValues.resize(nodeValues.size());
	activationFunction = activation_Function;
	bias = _bias;
}

Layer::Layer(int count, const ActivationFunction &activation_Function, bool _bias)
{
	nodeValues.resize(count);
	backNodeValues.resize(count);
	activationFunction = activation_Function;
	bias = _bias;
}

Layer::Layer(const Layer& layer)
{
	nodeValues = layer.nodeValues;
	backNodeValues = layer.backNodeValues;
	activationFunction = layer.activationFunction;
	bias = layer.bias;
}

Layer::~Layer()
{
	nodeValues.clear();
}

double Layer::GetNodeValue(int n) const
{
	if (n < 0)
	{
		printf("Error : %d번째 인자를 선택하였습니다.\n", n);
		return 0;
	}
	return nodeValues[n];
}

vector<double> Layer::GetNodeValue() const
{
	return nodeValues;
}

void Layer::SetNodeValue(int n, double value)
{
	nodeValues[n] = value;
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

double Layer::DActivate(double value)
{
	switch (activationFunction)
	{
	case ActivationFunction::Linear:
		return 1.0;
	case ActivationFunction::ReLU:
		if (value >= 0) { return 1.0; }
		else { return 0.0; }

	case ActivationFunction::Step:
		return 0.0;
	}

	return 1.0;
}

void Layer::SetBackNodeValue(int n, double value)
{
	backNodeValues[n] = value;
}

double Layer::GetBackNodeValue(int n) const
{
	return backNodeValues[n];
}
