#include "Optimizer.h"


GradientDescent::GradientDescent(double learning_Rate)
{
	learningRate = learning_Rate;
}

double GradientDescent::GetUpdateValue(double weight_value, double back_NodeValue, double prev_NodeValue)
{
	return weight_value - learningRate * (back_NodeValue * prev_NodeValue);
}