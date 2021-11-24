﻿#pragma once
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

	/**
	 * 노드 개수 가져오기.
	 *
	 * \return 노드 개수
	 */
	int GetNodeCount();

private:
	vector<double> nodeValues;
};