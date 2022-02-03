#pragma once
#include "ReinforcementLearning.h"
#include <map>
#include <string>


class QLearning : public ReinforcementLearning
{
public:
	/**
	 * csv ���Ϸ� ���º� �ൿ�� ���� ���� ���� �� ���� ǥ ����. (�� : �ൿ, �� : ����).
	 * ���¿� ���� ���Ǽҵ� ���� ����
	 *
	 * \param state_EndCondition : ���¿� ���� ���Ǽҵ� ���� ����
	 * \param next_StateTable : ���� ���� ǥ
	 * \param reward_Table : ����ǥ
	 */
	QLearning(string state_EndCondition, string next_StateTable, string reward_Table);
	/**
	 * csv ���Ϸ� ���º� �ൿ�� ���� ���� ���� �� ���� ǥ ����. (�� : �ൿ, �� : ����)
	 * ���� ���� ���Ǽҵ� ���� ����
	 * 
	 * \param reward_EndCondition : ���� ���� ���Ǽҵ� ���� ����
	 * \param next_StateTable : ���� ���� ǥ
	 * \param reward_Table : ����ǥ
	 */
	QLearning(int reward_EndCondition, string next_StateTable, string reward_Table);
	/**
	 * Ư�� ���¿��� �����Ͽ� �ϳ��� ���Ǽҵ带 ���� �� �н�.
	 * 
	 * \param starting_State : ���� ����
	 */
	void Learn(string starting_State);
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
	// �� ���º� �ൿ�� ���� ���� ǥ
	map<StateAction, double> rewardTable;
	// �� ���º� ������ �ൿ
	map<string, vector<string>> enableAction;

	void SetNextStateTable(const vector<vector<string>>& table);
	void SetRewardTable(const vector<vector<string>>& table);
};
