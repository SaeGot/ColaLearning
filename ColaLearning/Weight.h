﻿#pragma once
#include <vector>

using namespace std;


class Weight
{
public:
	/**
	 * 생성자.
	 * 
	 * \param weight_Values : 가중치
	 */
	Weight(vector<double> weight_Values);
	Weight(const Weight &weight);
	~Weight() {};

	/**
	 * 가중치값 가져오기.
	 * 
	 * \param n : 인덱스
	 * \return 가중치
	 */
	double GetWeight(int n);

private:
	vector<double> weightValues;
};

