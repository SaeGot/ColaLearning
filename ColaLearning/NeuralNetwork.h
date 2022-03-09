#pragma once
#include <vector>
#include "Weight.h"
#include "Layer.h"
#include "Optimizer.h"


using namespace std;

class NeuralNetwork
{
public:
	/**
	 * Layer 만으로 NeuralNetwork 생성.
	 * 
	 * \param _layers : 모든 층 (입력, 은닉, 출력 포함)
	 * \param layer_Count : 층 개수
	 * \param weight_InitialLimit : 가중치 초기값 상하한
	 */
	NeuralNetwork(const Layer* _layers, int layer_Count, double weight_InitialLimit = 1.0);
	/**
	 * Layer와 Weight로 NeuralNetwork 생성.
	 * 
	 * \param _layers : 모든 층 (입력, 은닉, 출력 포함)
	 * \param layer_Count : 층 개수
	 * \param _weights : 모든 가중치
	 */
	NeuralNetwork(const Layer* _layers, int layer_Count, const Weight* _weights);
	~NeuralNetwork();

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
	/**
	 * 학습.
	 * 
	 * \param input_Layers : 입력층
	 * \param target_Layers : 목표층
	 * \param optimizer : 최적화
	 * \param repeat : 반복 횟수
	 */
	void Learn(vector<Layer> input_Layers, vector<Layer> target_Layers, Optimizer* optimizer, int repeat = 1);

private:
	struct MinMax
	{
		double min = 0;
		double max = 0;
	};
	Layer* layers;
	Weight* weights;
	int layerCount;
	vector<MinMax> inputNodeMinMax;
	vector<MinMax> outputNodeMinMax;
	// 정규화 위한 최소 최대 설정 완료 여부
	bool minMaxSet;

	/**
	 * 모든 가중치 초기화.
	 * 
	 * \param weight_InitialLimit : 가중치 초기값 상하한
	 */
	void InitWeights(double weight_InitialLimit);
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
	 * Feed Forward 진행 (예측용).
	 * 
	 * \param input_Layer : 입력층
	 */
	void FeedForward(const Layer &input_Layer);
	/**
	 * Feed Forward 진행 (학습용).
	 * 
	 * \param input_Layer : 입력층
	 * \param target_Layer : 목표층
	 * \return 오차 (예측값 - 목표값)
	 */
	vector<double> FeedForward(const Layer& input_Layer, const Layer& target_Layer);
	/**
	 * 백노드 합.
	 * 
	 * \param next_Layer : 다음 층
	 * \param weight : 다음 층과 이전 층 사이의 가중치
	 * \param i : 합산 대상 노드 인덱스
	 * \return 
	 */
	double BackwardSum(const Layer& next_Layer, const Weight& weight, int i);
	/**
	 * 가중치 업데이트.
	 * 
	 * \param weight : 업데이트 대상 가중치
	 * \param prev_Layer : 이전 층
	 * \param i : 이전 층 노드 인덱스
	 * \param next_Layer : 다음 층
	 * \param j : 다음 층 노드 인덱스
	 * \param optimizer : 최적화 기법
	 */
	void UpdateWeight(Weight& weight, const Layer& prev_Layer, int i,
		Layer& next_Layer, int j, Optimizer* optimizer);
	/**
	 * 편향 가중치 업데이트.
	 * 
	 * \param weight : 업데이트 대상 가중치
	 * \param i : 이전 층 노드 인덱스
	 * \param next_Layer : 다음 층
	 * \param j : 다음 층 노드 인덱스
	 * \param optimizer : 최적화 기법
	 */
	void UpdateBiasWeight(Weight& weight, int i,
		Layer& next_Layer, int j, Optimizer* optimizer);
	/**
	 * 역전파.
	 * 
	 * \param errors : 오차
	 * \param optimizer : 최적화 기법
	 */
	void BackPropagation(vector<double> errors, Optimizer* optimizer);
	/**
	 * Normalize 위산 최소 최대 설정.
	 * 
	 * \param input_Layers : 입력 층
	 * \param target_Layers : 출력 층
	 */
	void SetMinMax(vector<Layer> input_Layers, vector<Layer> target_Layers);
};




