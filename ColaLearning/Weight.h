#pragma once
class Weight
{
public:
	/**
	 * 생성자.
	 * 
	 * \param weight : 가중치
	 */
	Weight(double weight);
	~Weight() {};

	/**
	 * 가중치값 가져오기.
	 * 
	 * \return : 가중치
	 */
	double GetWeight();

private:
	double MultipleValue;
};

