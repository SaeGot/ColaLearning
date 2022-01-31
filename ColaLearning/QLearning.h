#pragma once
#include "ReinforcementLearning.h"
#include <map>
#include <string>


class QLearning : public ReinforcementLearning
{
public:
	/**
	 * csv 파일로 상태별 행동에 따른 다음 상태 표 생성. (행 : 상태, 열 : 행동)
	 * 
	 * \param file_Name : 파일명
	 */
	QLearning(string file_Name);
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
	// ToDo 각 상태별 행동에 대해 어떤 상태가 되는지에 대한 표
	map<StateAction, string> nextStateTable;

	// i : 에피소드, j : 상태
	vector<vector<string>> stateList;
	// i : 에피소드, j : 행동
	vector<vector<string>> actionList;
};
