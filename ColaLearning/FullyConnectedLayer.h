#pragma once
#include "Layer.h"
using namespace std;


class FullyConnectedLayer : public Layer
{
public:
	/**
	 * ��尪���� Layer ����.
	 *
	 * \param node_Values : ��尪
	 * \param activation_Function : Ȱ�� �Լ�
	 * \param _bias : ����
	 */
	FullyConnectedLayer(map<Tensor, double> node_Values, const ActivationFunction& activation_Function = ActivationFunction::Linear,
		bool _bias = true) : Layer(node_Values, LayerType::FullyConnected, activation_Function, _bias) {}
	/**
	 * ��尪���� Layer ����. (1d ����)
	 *
	 * \param node_Values : ��尪
	 * \param activation_Function : Ȱ�� �Լ�
	 * \param _bias : ����
	 */
	FullyConnectedLayer(vector<double> node_Values, const ActivationFunction& activation_Function = ActivationFunction::Linear,
		bool _bias = true) : Layer(node_Values, LayerType::FullyConnected, activation_Function, _bias) {}
	/**
	 * ��� ������ Layer ����. (1d ����)
	 *
	 * \param node_Count : ��� ����
	 * \param activation_Function : Ȱ�� �Լ�
	 * \param _bias : ����
	 */
	FullyConnectedLayer(int node_Count, const ActivationFunction& activation_Function = ActivationFunction::Linear,
		bool _bias = true) : Layer(node_Count, LayerType::FullyConnected, activation_Function, _bias) {}
	/**
	 * ��� ������ Layer ����.
	 *
	 * \param x : x�� ��� ����
	 * \param y : x�� ��� ����
	 * \param channel : channel�� ��� ����
	 * \param activation_Function : Ȱ�� �Լ�
	 * \param _bias : ����
	 */
	FullyConnectedLayer(int x, int y, int channel = 1, const ActivationFunction& activation_Function = ActivationFunction::Linear,
		bool _bias = true) : Layer(x, y, channel, LayerType::FullyConnected, activation_Function, _bias) {}
	FullyConnectedLayer(const Layer& layer) : Layer(layer) {}
	FullyConnectedLayer(const ActivationFunction& activation_Function = ActivationFunction::Linear, bool _bias = true)
		: Layer(LayerType::FullyConnected, activation_Function, _bias) {}
	virtual ~FullyConnectedLayer();

	//virtual double ForwardSum(const Weight& weight, Tensor j);
};
