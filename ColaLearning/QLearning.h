#pragma once
#include "ReinforcementLearning.h"
#include <map>
#include <string>


class QLearning : public ReinforcementLearning
{
public:
	/**
	 * csv ���Ϸ� ���º� �ൿ�� ���� ���� ���� ǥ ����.
	 * 
	 * \param file_Name : ���ϸ�
	 */
	QLearning(string file_Name);
	void Learn();
	vector<string> GetBest();

protected:
	void TryStep();

private:
	struct StateAction
	{
		string state;
		string action;
	};
	// ToDo �� ���º� �ൿ�� ���� � ���°� �Ǵ����� ���� ǥ
	map<StateAction, string> nextState;

	// i : ���Ǽҵ�, j : ����
	vector<vector<string>> states;
	// i : ���Ǽҵ�, j : �ൿ
	vector<vector<string>> action;
};
