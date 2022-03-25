#pragma once
#include <map>
#include "Tensor.h"
using namespace std;


class Optimizer
{
public:
	/**
	 * ������Ʈ �� ��������.
	 * 
	 * \param weight_value : ����ġ ��
	 * \param back_NodeValue : ���� ��
	 * \param prev_NodeValue : ���� �� ��� ��
	 * \return ������Ʈ ��
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
