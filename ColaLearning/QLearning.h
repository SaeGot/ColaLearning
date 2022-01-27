#pragma once
#include "ReinforcementLearning.h"
#include <map>
#include <string>


class QLearning : public ReinforcementLearning
{
public:
	void Learn();
	vector<string> GetBest();

protected:
	void TryStep();

private:
	// ToDo �� ���º� �ൿ�� ���� � ���°� �Ǵ����� ���� ǥ

	// i : ���Ǽҵ�, j : ����
	vector<vector<string>> states;
	// i : ���Ǽҵ�, j : �ൿ
	vector<vector<string>> action;
};
