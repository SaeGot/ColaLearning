#include "Optimizer.h"
#include <math.h>


double Optimizer::Sigmoid(double value)
{
	return 1 / (1 + exp(-value));
}

SGD::SGD(double learning_Rate, double _momentum)
{
	learningRate = learning_Rate;
	momentum = _momentum;
	velocity = 0;
}

double SGD::GetUpdateValue(double weight_value, double back_NodeValue, double prev_NodeValue)
{
	velocity = momentum * velocity - learningRate * (back_NodeValue * prev_NodeValue);
	return weight_value + velocity;
}