#pragma once
class Weight
{
public:
	/**
	 * ������.
	 * 
	 * \param weight : ����ġ
	 */
	Weight(double weight);
	~Weight() {};

	/**
	 * ����ġ�� ��������.
	 * 
	 * \return : ����ġ
	 */
	double GetWeight();

private:
	double MultipleValue;
};

