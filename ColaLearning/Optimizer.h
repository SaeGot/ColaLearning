#pragma once


class Optimizer
{
public:

	virtual double GetUpdateValue(double weight_value, double back_NodeValue, double prev_NodeValue) = 0;

};

class GradientDescent : public Optimizer
{
public:
	GradientDescent(double learning_Rate);

	double GetUpdateValue(double weight_value, double back_NodeValue, double prev_NodeValue);

protected:
	double learningRate;
};
