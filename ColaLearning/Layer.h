#pragma once
#include <vector>
#include <map>
#include "Tensor.h"
using namespace std;


class Layer
{
public:
	enum class ActivationFunction
	{
		Linear,
		ReLU,
		Step,
		Softmax
	};
	enum class LayerType
	{
		FullyConnected,
		Convolution
	};
	/**
	 * 노드값으로 Layer 생성.
	 * 
	 * \param node_Values : 노드값
	 * \param layer_Type : 층 타입
	 * \param activation_Function : 활성 함수
	 * \param _bias : 편향
	 */
	Layer(map<Tensor, double> node_Values, LayerType layer_Type = LayerType::FullyConnected,
		const ActivationFunction& activation_Function = ActivationFunction::Linear, bool _bias = true);
	/**
	 * 노드값으로 Layer 생성. (1d 전용)
	 * 
	 * \param node_Values : 노드값
	 * \param layer_Type : 층 타입
	 * \param activation_Function : 활성 함수
	 * \param _bias : 편향
	 */
	Layer(vector<double> node_Values, LayerType layer_Type = LayerType::FullyConnected,
		const ActivationFunction &activation_Function = ActivationFunction::Linear, bool _bias = true);
	/**
	 * 노드 개수로 Layer 생성. (1d 전용)
	 * 
	 * \param node_Count : 노드 개수
	 * \param layer_Type : 층 타입
	 * \param activation_Function : 활성 함수
	 * \param _bias : 편향
	 */
	Layer(int node_Count, LayerType layer_Type = LayerType::FullyConnected,
		const ActivationFunction &activation_Function = ActivationFunction::Linear, bool _bias = true);
	/**
	 * 노드 개수로 Layer 생성.
	 * 
	 * \param x : x축 노드 개수
	 * \param y : x축 노드 개수
	 * \param channel : channel축 노드 개수
	 * \param layer_Type : 층 타입
	 * \param activation_Function : 활성 함수
	 * \param _bias : 편향
	 */
	Layer(int x, int y, int channel =  1, LayerType layer_Type = LayerType::FullyConnected,
		const ActivationFunction& activation_Function = ActivationFunction::Linear, bool _bias = true);
	Layer(const Layer &layer);
	Layer(LayerType layer_Type = LayerType::FullyConnected, const ActivationFunction& activation_Function = ActivationFunction::Linear, bool _bias = true);
	virtual ~Layer();

	/**
	 * 노드 값 가져오기.
	 *
	 * \param n : 인덱스
	 * \return 노드 값
	 */
	double GetNodeValue(Tensor n) const;
	/**
	 * 모든 노드 값 가져오기.
	 *
	 * \return 모든 노드 값
	 */
	map<Tensor, double> GetNodeValue();
	/**
	 * 노드 값 설정.
	 * 
	 * \param n : 노드 인덱스
	 * \param value : 노드 값
	 */
	void SetNodeValue(Tensor n, double value);
	/**
	 * 노드 개수 가져오기.
	 *
	 * \return 노드 개수
	 */
	int GetNodeCount() const;
	/**
	 * 모든 텐서 가져오기.
	 *
	 * \return 텐서
	 */
	vector<Tensor> GetTensorWithoutBias() const;
	/**
	 * 편향 여부 확인.
	 * 
	 * \return 편향 여부
	 */
	bool CheckBias() const;
	/**
	 * 활성화된 값 계산.
	 * 
	 * \param node_Value : 활성 전 노드 값
	 * \return 활성 후 노드 값
	 */
	double Activate(double node_Value);
	/**
	 * 활성함수의 미분 값.
	 * 
	 * \param node_Value : 활성함수 미분 전 노드 값
	 * \return 활성함수 미분 노드 값
	 */
	double Deactivate(double node_Value);
	/**
	 * BackPropagation을 위한 노드 값 설정.
	 * 
	 * \param n : 백노드 인덱스
	 * \param value : 백노드 값
	 */
	void SetBackNodeValue(Tensor n, double value);
	/**
	 * BackPropagation을 위한 노드 값 설정.
	 * 
	 * \param n : 백노드 인덱스
	 * \return 백노드 값
	 */
	double GetBackNodeValue(Tensor n) const;
	/**
	 * 활성 함수 가져오기.
	 * 
	 * \return 활성 함수
	 */
	ActivationFunction GetActivationFunction() const;
	/**
	 * 층 타입 가져오기.
	 * 
	 * \return 층 타입
	 */
	LayerType GetLayerType() const;
	/**
	 * 활성함수 설정.
	 * 
	 */
	void SetActivationFunction(ActivationFunction activation_Function);

protected:
	LayerType layerType;
	map<Tensor, double> nodeValues;
	map<Tensor, double> backNodeValues;
	ActivationFunction activationFunction;
	bool bias;

	/**
	 * 초기화.
	 *
	 */
	void Initialize();
};
