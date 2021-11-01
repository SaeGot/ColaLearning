#pragma once
#include <vector>

using namespace std;


class Node
{
public:
	/**
	 * ������.
	 * 
	 * \param node_Values : ��尪
	 */
	Node(vector<double> node_Values);
	~Node() {};

	/**
	 * ��� �� ��������.
	 * 
	 * \param n : �ε���
	 * \return ��� ��
	 */
	double GetNodeValue(int n);

private:
	vector<double> nodeValues;
};

