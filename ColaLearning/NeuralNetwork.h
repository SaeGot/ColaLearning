#pragma once
#include <vector>

#include "Weight.h"
#include "Layer.h"


using namespace std;

class NeuralNetwork
{
public:
	/**
	 * Layer 만으로 NeuralNetwork 생성.
	 * 
	 * \param _layers
	 */
	NeuralNetwork(vector<Layer> _layers);
	/**
	 * Layer와 Weight로 NeuralNetwork 생성.
	 * 
	 * \param _layers
	 * \param _weights
	 */
	NeuralNetwork(vector<Layer> _layers, vector<Weight> _weights);
	~NeuralNetwork() {};

	/**
	 * 예측 값들 계산.
	 * 
	 * \param layer : 입력층
	 * \return 예측 값
	 */
	vector<double> Predict(Layer layer);

private:
	vector<Layer> layers;
	vector<Weight> weights;

	/**
	 * 모든 가중치 초기화.
	 * 
	 */
	void InitWeights();
	/**
	 * 활성화 이전의 노드 값 계산.
	 * 
	 * \param layer : 이전 Layer
	 * \param weight : 이전 Layer와 다음 Layer 사이의 가중치
	 * \param j : 다음 Layer의 노드 인덱스
	 * \return 다음 노드 j 번째 값
	 */
	double Sum(const Layer &layer, const Weight &weight, int j);
};