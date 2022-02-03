#pragma once
#include "NeuralNetwork.h"
#include <string>


class ReinforcementLearning
{
public:
	enum class EpisodeEndCondition
	{
		State,
		Reward
	};

protected:
	struct SARS
	{
		string currentState;
		string action;
		double reward;
		string nextState;
	};
	EpisodeEndCondition episodeEndCondition;
	string stateEndCondition;
	int rewardEndCondition;
	string currentState;
	// i : 에피소드, j : 상태
	vector<vector<SARS>> sarsList;
};
