#pragma once
#include "ReinforcementLearning.h"
#include <map>
#include <string>


class QLearning : public ReinforcementLearning
{
public:
	/**
	 * csv 파일로 상태별 행동에 따른 다음 상태 및 보상 표 생성. (행 : 행동, 열 : 상태).
	 * 상태에 따른 에피소드 종료 조건
	 *
	 * \param state_EndCondition : 상태에 따른 에피소드 종료 조건
	 * \param next_StateTable : 다음 상태 표
	 * \param reward_Table : 보상표
	 */
	QLearning(string state_EndCondition, string next_StateTable, string reward_Table);
	/**
	 * csv 파일로 상태별 행동에 따른 다음 상태 및 보상 표 생성. (행 : 행동, 열 : 상태)
	 * 보상에 따른 에피소드 종료 조건
	 * 
	 * \param reward_EndCondition : 보상에 따른 에피소드 종료 조건
	 * \param next_StateTable : 다음 상태 표
	 * \param reward_Table : 보상표
	 */
	QLearning(int reward_EndCondition, string next_StateTable, string reward_Table);
	/**
	 * 특정 상태에서 시작하여 하나의 에피소드를 진행 및 학습.
	 * 
	 * \param starting_State : 시작 상태
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
	// 각 상태별 행동에 대해 어떤 상태가 되는지에 대한 표
	map<StateAction, string> nextStateTable;
	// 각 상태별 행동에 대한 보상 표
	map<StateAction, double> rewardTable;
	// 각 상태별 가능한 행동
	map<string, vector<string>> enableAction;

	void SetNextStateTable(const vector<vector<string>>& table);
	void SetRewardTable(const vector<vector<string>>& table);
};
