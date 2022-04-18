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
		// �� ���� ���� ������ �����ϴ��� ����. 0 ���ϴ� 1�� ����
		int interval;
		// ������
		float gamma;
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
