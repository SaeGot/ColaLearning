#pragma once
#include <vector>

using namespace std;


class Weight
{
public:
	/**
	 * ������.
	 * 
	 * \param weight_Values : ����ġ
	 */
	Weight(vector<double> weight_Values);
	~Weight() {};

	/**
	 * ����ġ�� ��������.
	 * 
	 * \param n : �ε���
	 * \return ����ġ
	 */
	double GetWeight(int n);

private:
	vector<double> weightValues;
};

