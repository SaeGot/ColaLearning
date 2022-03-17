#pragma once
#include "Layer.h"
using namespace std;


class FullyConnectedLayer : public Layer
{
public:
	/**
	 * 노드값으로 Layer 생성.
	 *
	 * \param node_Values : 노드값
	 * \param activation_Function : 활성 함수
	 * \param _bias : 편향
	 */
	FullyConnectedLayer(map<Tensor, double> node_Values, const ActivationFunction& activation_Function = ActivationFunction::Linear,
		bool _bias = true) : Layer(node_Values, LayerType::FullyConnected, activation_Function, _bias) {}
	/**
	 * 노드값으로 Layer 생성. (1d 전용)
	 *
	 * \param node_Values : 노드값
	 * \param activation_Function : 활성 함수
	 * \param _bias : 편향
	 */
	FullyConnectedLayer(vector<double> node_Values, const ActivationFunction& activation_Function = ActivationFunction::Linear,
		bool _bias = true) : Layer(node_Values, LayerType::FullyConnected, activation_Function, _bias) {}
	/**
	 * 노드 개수로 Layer 생성. (1d 전용)
	 *
	 * \param node_Count : 노드 개수
	 * \param activation_Function : 활성 함수
	 * \param _bias : 편향
	 */
	FullyConnectedLayer(int node_Count, const ActivationFunction& activation_Function = ActivationFunction::Linear,
		bool _bias = true) : Layer(node_Count, LayerType::FullyConnected, activation_Function, _bias) {}
	/**
	 * 노드 개수로 Layer 생성.
	 *
	 * \param x : x축 노드 개수
	 * \param y : x축 노드 개수
	 * \param channel : channel축 노드 개수
	 * \param activation_Function : 활성 함수
	 * \param _bias : 편향
	 */
	FullyConnectedLayer(int x, int y, int channel = 1, const ActivationFunction& activation_Function = ActivationFunction::Linear,
		bool _bias = true) : Layer(x, y, channel, LayerType::FullyConnected, activation_Function, _bias) {}
	FullyConnectedLayer(const Layer& layer) : Layer(layer) {}
	FullyConnectedLayer(const ActivationFunction& activation_Function = ActivationFunction::Linear, bool _bias = true)
		: Layer(LayerType::FullyConnected, activation_Function, _bias) {}
	virtual ~FullyConnectedLayer();

	//virtual double ForwardSum(const Weight& weight, Tensor j);
};
