#pragma once
#include "ReinforcementLearning.h"
#include <map>
#include <string>


class QLearning : public ReinforcementLearning
{
public:
	/**
	 * csv ���Ϸ� ���º� �ൿ�� ���� ���� ���� ǥ ����. (�� : �ൿ, �� : ����).
	 * ���¿� ���� ���Ǽҵ� ���� ����
	 *
	 * \param state_EndCondition : ���¿� ���� ���Ǽҵ� ���� ����
	 * \param file_Name : ���ϸ�
	 */
	QLearning(string state_EndCondition, string file_Name);
	/**
	 * csv ���Ϸ� ���º� �ൿ�� ���� ���� ���� ǥ ����. (�� : �ൿ, �� : ����)
	 * ���� ���� ���Ǽҵ� ���� ����
	 * 
	 * \param reward_EndCondition : ���� ���� ���Ǽҵ� ���� ����
	 * \param file_Name : ���ϸ�
	 */
	QLearning(int reward_EndCondition, string file_Name);
	void Learn(string state);
	void Action(string action = "");
	vector<string> GetBest();

protected:
	void Initialize(string state);

private:
	struct StateAction
	{
		string state;
		string action;

		bool operator<(const StateAction& rhs) const;
	};
	// �� ���º� �ൿ�� ���� � ���°� �Ǵ����� ���� ǥ
	map<StateAction, string> nextStateTable;
	// �� ���º� ������ �ൿ
	map<string, vector<string>> enableAction;
	// i : ���Ǽҵ�, j : ����
	vector<vector<string>> stateList;
	// i : ���Ǽҵ�, j : �ൿ
	vector<vector<string>> actionList;


	void SetTable(const vector<vector<string>>& table);
};
