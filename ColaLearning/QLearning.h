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
	 * ��� ���Ǽҵ� �ʱ�ȭ.
	 * 
	 */
	void Initialize();
	/**
	 * Ư�� ���¿��� �����Ͽ� �ϳ��� ���Ǽҵ带 ���� �� �н�.
	 * 
	 * \param starting_State : ���� ����
	 * \param discount_Factor : ������
	 * \param epsilon_Greedy
	 */
	void Learn(string starting_State, double discount_Factor, EpsilonGreedy& epsilon_Greedy);
	void Action(string action = "");
	vector<string> GetBest(string starting_State);

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
	// �� ���º� Q�� ǥ
	map<StateAction, double> QTable;

	void SetNextStateTable(const vector<vector<string>>& table);
	void SetRewardTable(const vector<vector<string>>& table);
	void UpdateQTable(double discount_Factor);
	string GetBestAction(string state);
	bool GetRandomPolicy(EpsilonGreedy& epsilon_Greedy, size_t step);
};
