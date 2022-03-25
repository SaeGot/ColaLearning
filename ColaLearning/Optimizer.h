#pragma once
#include <map>
#include "Tensor.h"
using namespace std;


class Optimizer
{
public:
	/**
	 * 업데이트 값 가져오기.
	 * 
	 * \param weight_value : 가중치 값
	 * \param back_NodeValue : 백노드 값
	 * \param prev_NodeValue : 이전 층 노드 값
	 * \return 업데이트 값
	 */
	virtual double GetUpdateValue(double weight_value, double back_NodeValue, double prev_NodeValue) = 0;

protected:

	double Sigmoid(double value);
};

class SGD : public Optimizer
{
public:
	SGD(double learning_Rate, double _momentum = 0.9);

	double GetUpdateValue(double weight_value, double back_NodeValue, double prev_NodeValue);

protected:
	double learningRate;
	double momentum;
	double velocity;
};
