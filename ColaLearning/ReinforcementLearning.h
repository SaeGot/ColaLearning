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
	struct EpsilonGreedy
	{
		float beginningValue;
		// 몇 스텝 마다 감가율 적용하는지 결정. 0 이하는 1로 적용
		int interval;
		// 감가율
		float gamma;
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
	double rewardEndCondition[2];
	string currentState;
	double cumulativeReward;
	// i : 에피소드, j : 상태
	vector<vector<SARS>> sarsList;
};
