#pragma once
#include <vector>

using namespace std;


class Node
{
public:
	/**
	 * 생성자.
	 * 
	 * \param node_Values : 노드값
	 */
	Node(vector<double> node_Values);
	~Node() {};

	/**
	 * 노드 값 가져오기.
	 * 
	 * \param n : 인덱스
	 * \return 노드 값
	 */
	double GetNodeValue(int n);

private:
	vector<double> nodeValues;
};

