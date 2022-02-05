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
		// true : ���ܸ��� ������ ����, false : ���Ǽҵ帶�� ������ ����
		bool discountPerStep;
		// �� interval���� ���� Ȥ�� ���Ǽҵ帶�� ������ �����ϴ��� ����. 0 ���ϴ� 1�� ����
		int interval;
		// ������
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
	// i : ���Ǽҵ�, j : ����
	vector<vector<SARS>> sarsList;
};
