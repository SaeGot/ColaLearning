#pragma once
#include <vector>


using namespace std;

class Weight
{
public:
	/**
	 * ������.
	 * 
	 * \param weight : ����ġ
	 */
	Weight(vector<double> weight_Values);
	~Weight() {};

	/**
	 * ����ġ�� ��������.
	 * 
	 * \return : ����ġ
	 */
	double GetWeight(int n);

private:
	vector<double> weightValues;
};

