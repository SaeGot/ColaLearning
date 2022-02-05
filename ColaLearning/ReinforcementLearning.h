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
	struct epsilonGreedy
	{
		float startValue;
		// true : 스텝마다 감가율 적용, false : 에피소드마다 감가율 적용
		bool discountPerStep;
		// 몇 interval마다 스텝 혹은 에피소드마다 감가율 적용하는지 결정. 0 이하는 1로 적용
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
	int rewardEndCondition;
	string currentState;
	// i : 에피소드, j : 상태
	vector<vector<SARS>> sarsList;
};
