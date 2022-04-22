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
	 * \param epsilon_Greeedy : epsilon greedy
	 */
	QLearning(string state_EndCondition, string next_StateTable, string reward_Table, double epsilon_Greeedy = 1);
	/**
	 * csv ���Ϸ� ���º� �ൿ�� ���� ���� ���� �� ���� ǥ ����. (�� : �ൿ, �� : ����)
	 * ���� ���� ���Ǽҵ� ���� ����
	 * 
	 * \param min_RewardEndCondition : ���� ���� ���Ǽҵ� ���� ���� (0 ������ �ּҰ�, 0�� ��� �� ������ ����)
	 * \param max_RewardEndCondition : ���� ���� ���Ǽҵ� ���� ���� (0 �ʰ��� �ִ밪)
	 * \param next_StateTable : ���� ���� ǥ
	 * \param reward_Table : ����ǥ
	 * \param epsilon_Greeedy : epsilon greedy
	 */
	QLearning(double min_RewardEndCondition, double max_RewardEndCondition, string next_StateTable, string reward_Table, double epsilon_Greeedy = 1);
	/**
	 * ��� ���Ǽҵ� �ʱ�ȭ.
	 * 
	 * \param epsilon_Greeedy : epsilon greedy
	 */
	void Initialize(double epsilon_Greeedy = 1);
	/**
	 * Ư�� ���¿��� �����Ͽ� �ϳ��� ���Ǽҵ带 ���� �� �н�.
	 * 
	 * \param starting_State : ���� ����
	 * \param discount_Factor : ������
	 * \param epsilon_Greedy : epsilon greedy
	 */
	void Learn(string starting_State, double discount_Factor, double epsilon_Greedy);
	/**
	 * �ൿ �����Ͽ� ���� ���·� ����.
	 * 
	 * \param action : �ൿ
	 */
	void Action(string action = "");
	/**
	 * ���� ���¿��� �����Ͽ� ���� ���� �ൿ���� ��������.
	 * 
	 * \param starting_State : ���� ����
	 * \return ���� �ൿ��
	 */
	vector<string> GetBest(string starting_State);
	/**
	 * ���� ���¿��� ���� ���� �ൿ ��������.
	 *
	 * \param state : ���� ����
	 * \return ���� �ൿ
	 */
	string GetBestAction(string state);
	/**
	 * ���� ���¿��� �ൿ ��������.
	 * 
	 * \param state : ���� ����
	 * \param random : ���� ����
	 * \return �ൿ
	 */
	string GetAction(string state, bool random);
	/**
	 * ���� ���� ����.
	 * 
	 */
	void SetCurrentState(string state);
	/**
	 * ���� ���� ��������.
	 * 
	 * \return ���� ����
	 */
	string GetCurrentState();
	/**
	 * Q Table �ҷ�����.
	 *
	 * \param file_Path : ���ϰ��
	 */
	void LoadQTable(string file_Path);
	/**
	 * Epsilon Greedy ����.
	 * 
	 * \param decay_Rate : ������
	 */
	void DecayEpsilonGreedy(double decay_Rate);
	void RemoveLastSART();

protected:
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
	map<StateAction, double> qTable;

	/**
	 * nextStateTable ����.
	 * 
	 * \param table : nextStateTable
	 */
	void SetNextStateTable(const vector<vector<string>>& table);
	/**
	 * rewardTable ����.
	 * 
	 * \param table : rewardTable
	 */
	void SetRewardTable(const vector<vector<string>>& table);
	/**
	 * QTable ����.
	 * 
	 * \param table : QTable
	 */
	void SetQTable(const vector<vector<string>>& table);
	/**
	 * QTable ������Ʈ.
	 * 
	 * \param discount_Factor : ������
	 */
	virtual void UpdateQTable(double discount_Factor);
	/**
	 * ���� ���� ���� ���� ��������.
	 * 
	 * \param cumulative_Reward : ���� ����
	 * \return ���� ���� ���� ����
	 */
	bool CheckRewardEndCondition(double cumulative_Reward);
};
