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
	EpisodeEndCondition episodeEndCondition;
	string stateEndCondition;
	int rewardEndCondition;
	string currentState;
	int currentReward;
};
