#pragma once
#include <vector>


using namespace std;

enum class ActivationFunction
{
	Linear,
	ReLU,
	Step
};

class Layer
{
public:
	/**
	 * 노드값으로 Layer 생성.
	 * 
	 * \param node_Values : 노드값
	 * \param activation_Function : 활성 함수
	 * \param _bias : 편향
	 */
	Layer(vector<double> node_Values, ActivationFunction activation_Function = ActivationFunction::Linear, bool _bias = false);
	/**
	 * 노드 개수로 Layer 생성.
	 * 
	 * \param count : 노드 개수
	 * \param activation_Function : 활성 함수
	 * \param _bias : 편향
	 */
	Layer(int count, ActivationFunction activation_Function = ActivationFunction::Linear, bool _bias = false);
	Layer(const Layer& layer);
	~Layer();

	/**
	 * 노드 값 가져오기.
	 *
	 * \param index : 인덱스
	 * \return 노드 값
	 */
	double GetNodeValue(int index) const;
	/**
	 * 모든 노드 값 가져오기.
	 * 
	 * \return 모든 노드 값
	 */
	vector<double> GetNodeValue() const;
	/**
	 * 노드 값 설정.
	 * 
	 * \param index : 노드 인덱스
	 * \param value : 가중치 값
	 */
	void SetNodeValue(int index, double value);
	/**
	 * 노드 값 초기화.
	 * 
	 */
	void InitNodeValue();
	/**
	 * 노드 개수 가져오기.
	 *
	 * \return 노드 개수
	 */
	int GetNodeCount() const;
	/**
	 * 편향 여부 확인.
	 * 
	 * \return 편향 여부
	 */
	bool CheckBias() const;
	/**
	 * 활성화된 값 계산.
	 * 
	 * \param value : 활성화 전 값
	 */
	double Activate(double value);

private:
	vector<double> nodeValues;
	bool bias;
	ActivationFunction activationFunction;
};
