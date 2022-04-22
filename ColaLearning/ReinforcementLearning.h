#pragma once
#include <string>
#include <vector>
using namespace std;


class ReinforcementLearning
{
public:
	enum class EpisodeEndCondition
	{
		State,
		Reward
	};
	/**
	 * epsilon greedy 감소.
	 * 
	 * \param decay : 감소 비율
	 */
	void DecayEpsilonGreedy(double decay);
	double GetEpsilonGreedy() { return epsilonGreeedy; }

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
	// 0 ~ 1값, 1 = 완전 래덤, 0 = 탐욕
	double epsilonGreeedy;
};
