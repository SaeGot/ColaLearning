#pragma once
#include <vector>
#include <map>
#include "Layer.h"
#include "Tensor.h"
using namespace std;


enum class InitWeight
{
	RamdomUniform,
	He,
	Xavier
};

class Weight
{
public:
	/**
	 * 가중치값으로 Weight 생성. (1D 전용)
	 * 
	 * \param weight_Values : 가중치
	 * \param bias : 이전 층 편향 유무
	 */
	Weight(vector<vector<double>> weight_Values, bool previous_Bias = true);
	/**
	 * 이전 층, 다음 층 노드 개수로 Weight 생성. (편향 무조건 포함)
	 * 
	 * \param previous_NodeCountWithBias : 편항 포함한 이전층 노드 개수
	 * \param next_NodeCount : 다음층 노드 개수
	 * \param init_Weight : 초기화 방법
	 * \param initial_Limit : 가중치 초기값 상하한
	 */
	Weight(int input_NodeCountWithBias, int output_NodeCount, InitWeight init_Weight, double initial_Limit);
	/**
	 * 이전 층, 다음 층으로 Weight 생성.
	 * 
	 * \param previous_Layer : 이전 층
	 * \param next_Layer : 다음 층
	 * \param init_Weight : 초기화 방법
	 * \param initial_Limit : 가중치 초기값 상하한
	 */
	Weight(Layer* previous_Layer, Layer* next_Layer, InitWeight init_Weight, double initial_Limit);
	Weight(const Weight &weight);
	Weight();
	~Weight();

	/**
	 * 가중치값 가져오기.
	 * 
	 * \param i : 이전 층 노드 인덱스
	 * \param j : 다음 층 노드 인덱스
	 * \return 가중치
	 */
	double GetWeight(Tensor i, Tensor j) const;
	/**
	 * 다음층 j 노드와 연결된 모든 가중치 가져오기.
	 * 
	 * \param j : 다음층 노드 인덱스
	 * \return 가중치
	 */
	map<TensorConnection, double> GetJWeightValues(Tensor j) const;
	/**
	 * 가중치 업데이트.
	 *
	 * \param i : 이전 층 노드 인덱스
	 * \param j : 다음 층 노드 인덱스
	 * \param value : 업데이트할 가중치 값
	 */
	void UpdateWeight(Tensor i, Tensor j, double value);

private:
	map<TensorConnection, double> weightValues;
	bool previousBias;

	/**
	 * 가중치 초기화.
	 * 
	 * \param init_Weight : 초기화 방법
	 * \param input_NodeCountWithBias : 편향 포함 입력 노드 개수
	 * \param int output_NodeCount : 출력 노드 개수
	 * \param limit : 상하한 값
	 * \return 가중치 초기화 값
	 */
	double Initialize(InitWeight init_Weight, int input_NodeCountWithBias, int output_NodeCount, double limit);
	/**
	 * 가중치 모두 제거(초기화).
	 * 
	 */
	void Initialize();
};

