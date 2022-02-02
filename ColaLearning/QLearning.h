#pragma once
#include "ReinforcementLearning.h"
#include <map>
#include <string>


class QLearning : public ReinforcementLearning
{
public:
	/**
	 * csv 파일로 상태별 행동에 따른 다음 상태 표 생성. (행 : 행동, 열 : 상태).
	 * 상태에 따른 에피소드 종료 조건
	 *
	 * \param state_EndCondition : 상태에 따른 에피소드 종료 조건
	 * \param file_Name : 파일명
	 */
	QLearning(string state_EndCondition, string file_Name);
	/**
	 * csv 파일로 상태별 행동에 따른 다음 상태 표 생성. (행 : 행동, 열 : 상태)
	 * 보상에 따른 에피소드 종료 조건
	 * 
	 * \param reward_EndCondition : 보상에 따른 에피소드 종료 조건
	 * \param file_Name : 파일명
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
	// 각 상태별 행동에 대해 어떤 상태가 되는지에 대한 표
	map<StateAction, string> nextStateTable;
	// 각 상태별 가능한 행동
	map<string, vector<string>> enableAction;
	// i : 에피소드, j : 상태
	vector<vector<string>> stateList;
	// i : 에피소드, j : 행동
	vector<vector<string>> actionList;


	void SetTable(const vector<vector<string>>& table);
};
