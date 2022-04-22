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
	 * epsilon greedy ����.
	 * 
	 * \param decay : ���� ����
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
	// i : ���Ǽҵ�, j : ����
	vector<vector<SARS>> sarsList;
	// 0 ~ 1��, 1 = ���� ����, 0 = Ž��
	double epsilonGreeedy;
};
