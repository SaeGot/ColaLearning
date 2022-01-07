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
	 * \param input_Layer : 입력층
	 * \return 예측 값
	 */
	vector<double> Predict(Layer input_Layer);
	/**
	 * 오차 가져오기.
	 * 
	 * \param input_Layer : 입력층
	 * \param target_Layer : 목표층
	 * \return 오차 (예측값 - 목표값)
	 */
	vector<double> GetError(Layer input_Layer, Layer target_Layer);

	void Learn(vector<Layer> input_Layers, vector<Layer> target_Layers);

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
	double ForwardSum(const Layer &layer, const Weight &weight, int j);
	/**
	 * Feed Forward 진행.
	 * 
	 * \param input_Layer : 입력층
	 */
	void FeedForward(const Layer &input_Layer);
	/**
	 * Feed Forward 진행.
	 * 
	 * \param input_Layer : 입력층
	 * \param target_Layer : 목표층
	 * \return 오차 (예측값 - 목표값)
	 */
	vector<double> FeedForward(const Layer& input_Layer, const Layer& target_Layer);

	double BackwardSum(const Layer& layer, const Weight& weight, int i);
	void BackPropagation(const Layer &target_Layer, vector<double> errors);
	void UpdateWeight(Weight &weight, const Layer& prev_Layer, int i,
		Layer &next_Layer, int j, double error);
	void UpdateBiasWeight(Weight& weight, int i,
		Layer& next_Layer, int j, double error);
};