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
	// ToDo 각 상태별 행동에 대해 어떤 상태가 되는지에 대한 표

	// i : 에피소드, j : 상태
	vector<vector<string>> states;
	// i : 에피소드, j : 행동
	vector<vector<string>> action;
};
