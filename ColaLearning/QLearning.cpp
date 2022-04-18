#include "QLearning.h"
#include "FileManager.h"
#include <random>


QLearning::QLearning(string state_EndCondition, string next_StateTable, string reward_Table, double epsilon_Greeedy)
{
	episodeEndCondition = EpisodeEndCondition::State;
	stateEndCondition = state_EndCondition;

	FileManager file;
	vector<vector<string>> next_state_table = file.GetTable(next_StateTable);
	SetNextStateTable(next_state_table);
	vector<vector<string>> reward_table = file.GetTable(reward_Table);
	SetRewardTable(reward_table);

	Initialize(epsilon_Greeedy);
}

QLearning::QLearning(double min_RewardEndCondition, double max_RewardEndCondition, string next_StateTable, string reward_Table, double epsilon_Greeedy)
{
	// ToDo 예외 처리
	episodeEndCondition = EpisodeEndCondition::Reward;
	rewardEndCondition[0] = min_RewardEndCondition;
	rewardEndCondition[1] = max_RewardEndCondition;

	FileManager file;
	vector<vector<string>> next_state_table = file.GetTable(next_StateTable);
	SetNextStateTable(next_state_table);
	vector<vector<string>> reward_table = file.GetTable(reward_Table);
	SetRewardTable(reward_table);

	Initialize(epsilon_Greeedy);
}

void QLearning::Learn(string starting_State, double discount_Factor, EpsilonGreedy& epsilon_Greedy)
{
	// ToDo 예외 처리
	if (epsilon_Greedy.interval <= 0)
	{
		epsilon_Greedy.interval = 1;
	}
	if (epsilon_Greedy.gamma < 0 || epsilon_Greedy.gamma > 1)
	{
		epsilon_Greedy.gamma = 1;
	}

	currentState = starting_State;
	cumulativeReward = 0;

	if (sarsList.back().size() > 0)
	{
		vector<SARS> sars;
		sarsList.push_back(sars);
	}

	if (episodeEndCondition == EpisodeEndCondition::State)
	{
		while (currentState != stateEndCondition)
		{
			string action = "";
			if (!GetRandomPolicy(epsilon_Greedy, sarsList.back().size() + 1))
			{
				action = GetBestAction(currentState);
			}
			Action(action);
			UpdateQTable(discount_Factor);
		}
	}
	else if (episodeEndCondition == EpisodeEndCondition::Reward)
	{
		while (CheckRewardEndCondition(cumulativeReward))
		{
			string action = "";
			if (!GetRandomPolicy(epsilon_Greedy, sarsList.back().size() + 1))
			{
				action = GetBestAction(currentState);
			}
			Action(action);
			UpdateQTable(discount_Factor);
		}
	}
}

void QLearning::Initialize(double epsilon_Greeedy)
{
	epsilonGreeedy = epsilon_Greeedy;
	currentState = "";
	for (vector<SARS>& sars : sarsList)
	{
		sars.clear();
	}
	sarsList.clear();
	vector<SARS> sars;
	sarsList.push_back(sars);
	
	qTable.clear();
	for (const pair<StateAction, double>& reward : rewardTable)
	{
		qTable.insert({ reward.first, 0.0 });
	}
}

void QLearning::Action(string action)
{
	string next_state;
	StateAction state_action;
	if (action == "")
	{
		const vector<string>& enable_action = enableAction[currentState];

		random_device rd;
		mt19937_64 gen(rd());
		int action_count = static_cast<int>(enable_action.size() - 1);
		uniform_int_distribution<int> random_number(0, action_count);

		action = enable_action[random_number(gen)];
	}
	state_action = { currentState, action };
	double reward = rewardTable[state_action];
	next_state = nextStateTable[state_action];

	SARS sars = { currentState, action, reward, next_state };
	sarsList.back().push_back(sars);
	// 다음 상태로 전이
	currentState = next_state;
	cumulativeReward += reward;
}

vector<string> QLearning::GetBest(string starting_State)
{
	vector<string> best_way;

	string current_state = starting_State;
	string next_state;
	double cumulative_reward = 0;

	if (episodeEndCondition == EpisodeEndCondition::State)
	{
		while (current_state != stateEndCondition)
		{
			string max_q_state;
			double max_q;
			bool first_action = true;
			for (const string& action : enableAction[current_state])
			{
				StateAction state_action = { current_state, action };
				if (first_action)
				{
					next_state = nextStateTable[state_action];
					max_q = qTable[state_action];
					first_action = false;
				}
				else
				{
					if (qTable[state_action] > max_q)
					{
						next_state = nextStateTable[state_action];
						max_q = qTable[state_action];
					}
				}
			}
			best_way.push_back(next_state);
			current_state = next_state;
		}
	}
	else if (episodeEndCondition == EpisodeEndCondition::Reward)
	{
		while (CheckRewardEndCondition(cumulative_reward))
		{
			string max_q_state;
			double max_q;
			bool first_action = true;
			double reward = 0;
			for (const string& action : enableAction[current_state])
			{
				StateAction state_action = { current_state, action };
				if (first_action)
				{
					next_state = nextStateTable[state_action];
					max_q = qTable[state_action];
					reward = rewardTable[state_action];
					first_action = false;
				}
				else
				{
					if (qTable[state_action] > max_q)
					{
						next_state = nextStateTable[state_action];
						max_q = qTable[state_action];
						reward = rewardTable[state_action];
					}
				}
			}
			best_way.push_back(next_state);
			current_state = next_state;
			cumulative_reward += reward;
		}
	}

	return best_way;
}

string QLearning::GetBestAction(string state)
{
	// 첫번째
	bool only_one_best_action = true;
	vector<string> best_actions;
	string action = enableAction[state][0];
	best_actions.push_back(action);
	string best_action = action;
	StateAction state_action = { state, best_action };
	double best_q = qTable[state_action];
	// 두번째 부터
	for (int n = 1; n < enableAction[state].size(); n++)
	{
		action = enableAction[state][n];
		state_action = { state, action };
		if (qTable[state_action] > best_q)
		{
			best_action = action;
			best_q = qTable[state_action];
			best_actions.clear();
			best_actions.push_back(best_action);
			only_one_best_action = true;
		}
		else if (qTable[state_action] == best_q)
		{
			best_actions.push_back(action);
			only_one_best_action = false;
		}
	}

	// 최고 Q값 행동 가져오기
	random_device rd;
	mt19937_64 gen(rd());
	uniform_int_distribution<int> random_value(0, best_actions.size() - 1);
	int random_num = random_value(gen);
	best_action = best_actions[random_num];

	return best_action;
}

string QLearning::GetAction(string state, bool random)
{
	string action;

	double epsilon_Greedy;

	if (random)
	{
		epsilon_Greedy = 1;
	}
	else
	{
		epsilon_Greedy = epsilonGreeedy;
	}

	random_device rd;
	mt19937_64 gen(rd());
	uniform_real_distribution<double> random_value(0, 1);
	double random_num = random_value(gen);
	// 탐욕
	if (random_num >= epsilon_Greedy)
	{
		action = GetBestAction(state);
	}
	// 랜덤
	else
	{
		uniform_int_distribution<int> random_index(0, enableAction[state].size() - 1);
		int index = random_index(gen);
		action = enableAction[state][index];
	}

	return action;
}

void QLearning::SetCurrentState(string state)
{
	currentState = state;
}

string QLearning::GetCurrentState()
{
	return currentState;
}

void QLearning::LoadQTable(string file_Path)
{
	FileManager file;
	vector<vector<string>> reward_table = file.GetTable(file_Path);
	qTable.clear();
	SetQTable(reward_table);
}

void QLearning::RemoveLastSART()
{
	sarsList.back().erase(sarsList.back().end() - 1);
}

bool QLearning::StateAction::operator<(const StateAction& rhs) const
{
	if (state != rhs.state)
	{
		return state < rhs.state;
	}
	
	return action < rhs.action;
}

void QLearning::SetNextStateTable(const vector<vector<string>>& table)
{
	vector<string> first_row;
	first_row.push_back("");
	for (int j = 1; j < table[0].size(); j++)
	{
		first_row.push_back(table[0][j]);
	}
	// i : row, j : col
	for (int i = 1; i < table.size(); i++)
	{
		const vector<string>& row = table[i];
		vector<string> action;
		for (int j = 1; j < row.size(); j++)
		{
			StateAction state_action = { row[0], first_row[j] };
			if (row[j] != "")
			{
				nextStateTable.insert({ state_action, row[j] });
				action.push_back(first_row[j]);
			}
		}
		enableAction.insert({ row[0], action });
	}
}

void QLearning::SetRewardTable(const vector<vector<string>>& table)
{
	vector<string> first_row;
	first_row.push_back("");
	for (int j = 1; j < table[0].size(); j++)
	{
		first_row.push_back(table[0][j]);
	}
	// i : row, j : col
	for (int i = 1; i < table.size(); i++)
	{
		const vector<string>& row = table[i];
		vector<string> action;
		for (int j = 1; j < row.size(); j++)
		{
			StateAction state_action = { row[0], first_row[j] };
			if (row[j] != "")
			{
				rewardTable.insert({ state_action, stod(row[j]) });
			}
			else
			{
				rewardTable.insert({ state_action, 0.0 });
			}
		}
	}
}

void QLearning::SetQTable(const vector<vector<string>>& table)
{
	vector<string> first_row;
	first_row.push_back("");
	for (int j = 1; j < table[0].size(); j++)
	{
		first_row.push_back(table[0][j]);
	}
	// i : row, j : col
	for (int i = 1; i < table.size(); i++)
	{
		const vector<string>& row = table[i];
		vector<string> action;
		for (int j = 1; j < row.size(); j++)
		{
			StateAction state_action = { row[0], first_row[j] };
			if (row[j] != "")
			{
				qTable.insert({ state_action, stod(row[j]) });
			}
			else
			{
				qTable.insert({ state_action, 0.0 });
			}
		}
	}
}

void QLearning::UpdateQTable(double discount_Factor)
{
	SARS sars = sarsList.back().back();
	StateAction state_action = { sars.currentState, sars.action };
	double reward = sars.reward;
	double max_q = 0;
	StateAction next_state_action;
	bool first_action = true;
	for (const string& next_next_action : enableAction[sars.nextState])
	{
		next_state_action = { sars.nextState, next_next_action };
		if (first_action)
		{
			max_q = qTable[next_state_action];
			first_action = false;
		}
		else
		{
			max_q = max(max_q, qTable[next_state_action]);
		}
	}
	qTable[state_action] = reward + (discount_Factor * max_q);
}

bool QLearning::GetRandomPolicy(EpsilonGreedy& epsilon_Greedy, size_t step)
{
	random_device rd;
	mt19937_64 gen(rd());
	uniform_real_distribution<double> random_value(0, 1);

	if (step % epsilon_Greedy.interval == 0)
	{
		epsilon_Greedy.beginningValue *= epsilon_Greedy.gamma;
	}

	if (epsilon_Greedy.beginningValue <= 0)
	{
		return false;
	}
	else if (epsilon_Greedy.beginningValue >= random_value(gen))
	{
		return true;
	}
	return false;
}

bool QLearning::CheckRewardEndCondition(double cumulative_Reward)
{
	if (rewardEndCondition[0] == 0)
	{
		return cumulative_Reward < rewardEndCondition[1];
	}
	else
	{
		return cumulative_Reward > rewardEndCondition[1] && cumulative_Reward < rewardEndCondition[1];
	}
}
