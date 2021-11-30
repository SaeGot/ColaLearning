#pragma once
#include <vector>


using namespace std;

class Layer
{
public:
	/**
	 * 노드값으로 Layer 생성.
	 * 
	 * \param node_Values : 노드값
	 * \param _bias : 편향
	 */
	Layer(vector<double> node_Values, bool _bias = false);
	/**
	 * 노드 개수로 Layer 생성.
	 * 
	 * \param count : 노드 개수
	 * \param _bias : 편향
	 */
	Layer(int count, bool _bias = false);
	Layer(const Layer& layer);
	~Layer() {};

	/**
	 * 노드 값 가져오기.
	 *
	 * \param index : 인덱스
	 * \return 노드 값
	 */
	double GetNodeValue(int index) const;
	/**
	 * 모든 노드 값 가져오기.
	 * 
	 * \return 모든 노드 값
	 */
	vector<double> GetNodeValue() const;
	/**
	 * 노드 값 설정.
	 * 
	 * \param index : 인덱스
	 * \param value : 가중치 값
	 */
	void SetNodeValue(int index, double value);
	/**
	 * 노드 값 초기화.
	 * 
	 */
	void InitNodeValue();
	/**
	 * 노드 개수 가져오기.
	 *
	 * \return 노드 개수
	 */
	int GetNodeCount() const;
	/**
	 * 편향 여부 확인.
	 * 
	 * \return 편향 여부
	 */
	bool CheckBias() const;

private:
	vector<double> nodeValues;
	bool bias;
};