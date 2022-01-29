#pragma once
#include "ReinforcementLearning.h"
#include <map>
#include <string>


class QLearning : public ReinforcementLearning
{
public:
	/**
	 * csv 파일로 상태별 행동에 따른 다음 상태 표 생성.
	 * 
	 * \param file_Name : 파일명
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
	// ToDo 각 상태별 행동에 대해 어떤 상태가 되는지에 대한 표
	map<StateAction, string> nextState;

	// i : 에피소드, j : 상태
	vector<vector<string>> states;
	// i : 에피소드, j : 행동
	vector<vector<string>> action;
};
