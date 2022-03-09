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
	Layer(vector<double> node_Values, const ActivationFunction &activation_Function = ActivationFunction::Linear, bool _bias = true);
	/**
	 * 노드 개수로 Layer 생성.
	 * 
	 * \param count : 노드 개수
	 * \param activation_Function : 활성 함수
	 * \param _bias : 편향
	 */
	Layer(int count, const ActivationFunction &activation_Function = ActivationFunction::Linear, bool _bias = true);
	Layer(const Layer &layer);
	Layer(const ActivationFunction& activation_Function = ActivationFunction::Linear, bool _bias = true);
	~Layer();

	/**
	 * 노드 값 가져오기.
	 *
	 * \param n : 인덱스
	 * \return 노드 값
	 */
	double GetNodeValue(int n) const;
	/**
	 * 모든 노드 값 가져오기.
	 * 
	 * \return 모든 노드 값
	 */
	vector<double> GetNodeValue() const;
	/**
	 * 노드 값 설정.
	 * 
	 * \param n : 노드 인덱스
	 * \param value : 노드 값
	 */
	void SetNodeValue(int n, double value);
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
	 * \return 활성 값
	 */
	double Activate(double value);
	/**
	 * 활성함수의 미분 값.
	 * 
	 * \param value : 활성함수 미분의 변수
	 * \return 활성함수 미분 값
	 */
	double DActivate(double value);
	/**
	 * BackPropagation을 위한 노드 값 설정.
	 * 
	 * \param n : 백노드 인덱스
	 * \param value : 백노드 값
	 */
	void SetBackNodeValue(int n, double value);
	/**
	 * BackPropagation을 위한 노드 값 설정.
	 * 
	 * \param n : 백노드 인덱스
	 * \return 백노드 값
	 */
	double GetBackNodeValue(int n) const;
	/**
	 * 활성 함수 가져오기.
	 * 
	 * \return : 활성 함수
	 */
	ActivationFunction GetActivationFunction() const;

private:
	vector<double> nodeValues;
	vector<double> backNodeValues;
	ActivationFunction activationFunction;
	bool bias;
};
