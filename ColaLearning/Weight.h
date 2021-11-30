#pragma once
#include <vector>


using namespace std;

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
	 * \param prev_node_count : 이전 층 노드 개수
	 * \param next_node_count : 다음 층 노드 개수
	 */
	Weight(int prev_node_count, int next_node_count);
	Weight(const Weight &weight);
	~Weight() {};

	/**
	 * 가중치값 가져오기.
	 * 
	 * \param i : 이전 층 노드 인덱스
	 * \param j : 다음 층 노드 인덱스
	 * \return 
	 */
	double GetWeight(int i, int j) const;

private:
	vector<vector<double>> weightValues;
};

