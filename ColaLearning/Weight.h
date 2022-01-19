#pragma once
#include <vector>


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
	 * 가중치값으로 Weight 생성.
	 * 
	 * \param weight_Values : 가중치
	 */
	Weight(vector<vector<double>> weight_Values);
	/**
	 * 이전 층, 다음 층 노드 개수로 Weight 생성.
	 * 
	 * \param input_NodeCountWithBias : 편항 포함한 입력 노드 개수
	 * \param output_NodeCount : 출력 노드 개수
	 * \param init_Weight : 초기화 방법
	 */
	Weight(int input_NodeCountWithBias, int output_NodeCount, InitWeight init_Weight);
	Weight(const Weight &weight);
	~Weight();

	/**
	 * 가중치값 가져오기.
	 * 
	 * \param i : 이전 층 노드 인덱스
	 * \param j : 다음 층 노드 인덱스
	 * \return 
	 */
	double GetWeight(int i, int j) const;
	/**
	 * 가중치 업데이트.
	 *
	 * \param i : 이전 층 노드 인덱스
	 * \param j : 다음 층 노드 인덱스
	 * \param value
	 */
	void UpdateWeight(int i, int j, double value);

private:
	vector<vector<double>> weightValues;

	/**
	 * 가중치 초기화.
	 * 
	 * \param init_Weight : 초기화 방법
	 * \param input_NodeCountWithBias : 편향 포함 입력 노드 개수
	 * \param int output_NodeCount : 출력 노드 개수
	 * \param limit : 상하한 값
	 * \return 가중치 초기화 값
	 */
	double Initialize(InitWeight init_Weight, int input_NodeCountWithBias, int output_NodeCount, double limit = 1.0);
};

