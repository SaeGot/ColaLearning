#include "Layer.h"


Layer::Layer(map<Tensor, double> node_Values, const ActivationFunction& activation_Function, bool _bias)
{
	Initialize();
	nodeValues = node_Values;
	activationFunction = activation_Function;
	bias = _bias;
}

Layer::Layer(vector<double> node_Values, const ActivationFunction &activation_Function, bool _bias)
{
	Initialize();
	for (int n = 0; n < node_Values.size(); n++)
	{
		nodeValues.emplace(Tensor(n), node_Values[n]);
		backNodeValues.emplace(Tensor(n), 0);
	}
	activationFunction = activation_Function;
	bias = _bias;
}

Layer::Layer(int node_Count, const ActivationFunction &activation_Function, bool _bias)
{
	Initialize();
	for (int n = 0; n < node_Count; n++)
	{
		nodeValues.emplace(Tensor(n), 0);
		backNodeValues.emplace(Tensor(n), 0);
	}
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

Layer::Layer(const ActivationFunction& activation_Function, bool _bias)
{
	activationFunction = activation_Function;
	bias = _bias;
}

Layer::~Layer()
{
	Initialize();
}

double Layer::GetNodeValue(Tensor n) const
{
	/*
	if (n < 0)
	{
		printf("Error : %d번째 인자를 선택하였습니다.\n", n);
		return 0;
	}
	*/
	return nodeValues.at(Tensor(n));
}

vector<double> Layer::GetNodeValue() const
{
	vector<double> values;
	for (int n = 0; n < nodeValues.size(); n++)
	{
		values.push_back( nodeValues.at(Tensor(n)) );
	}

	return values;
}

void Layer::SetNodeValue(Tensor n, double value)
{
	nodeValues[n] = value;
}

int Layer::GetNodeCount() const
{
	return static_cast<int>(nodeValues.size());
}

vector<Tensor> Layer::GetTensorWithoutBias() const
{
	vector<Tensor> tensors;
	for (const pair<Tensor, double>& node_value : nodeValues)
	{
		if (node_value.first != Tensor::GetBias())
		{
			tensors.push_back(node_value.first);
		}
	}

	return tensors;
}

bool Layer::CheckBias() const
{
	return bias;
}

double Layer::Activate(double node_Value)
{
	switch (activationFunction)
	{
	case ActivationFunction::ReLU:
		if (node_Value >= 0) { return node_Value; }
		else { return 0.0; }

	case ActivationFunction::Step:
		if (node_Value >= 0) { return 1.0; }
		else { return 0.0; }
	}

	return node_Value;
}

double Layer::Deactivate(double node_Value)
{
	switch (activationFunction)
	{
	case ActivationFunction::Linear:
		return 1.0;
	case ActivationFunction::ReLU:
		if (node_Value >= 0) { return 1.0; }
		else { return 0.0; }

	case ActivationFunction::Step:
		return 0.0;
	}

	return 1.0;
}

void Layer::SetBackNodeValue(Tensor n, double value)
{
	backNodeValues.at(Tensor(n)) = value;
}

double Layer::GetBackNodeValue(Tensor n) const
{
	return backNodeValues.at(Tensor(n));
}

Layer::ActivationFunction Layer::GetActivationFunction() const
{
	return activationFunction;
}

Layer::LayerType Layer::GetLayerType() const
{
	return layerType;
}

void Layer::Initialize()
{
	nodeValues.clear();
	backNodeValues.clear();
}
