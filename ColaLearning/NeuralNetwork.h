#pragma once
#include <vector>

#include "Weight.h"
#include "Layer.h"


using namespace std;

class NeuralNetwork
{
public:
	NeuralNetwork(vector<Layer> _layers);
	NeuralNetwork(vector<Layer> _layers, vector<Weight> _weights);
	~NeuralNetwork() {};

	vector<double> Predict();

private:
	vector<Layer> layers;
	vector<Weight> weights;

	void InitWeights();
	double Sum(const Layer &layer, const Weight &weight, int j);
	double Activate(double value);
};