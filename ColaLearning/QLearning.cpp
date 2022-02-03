#include "QLearning.h"
#include "FileManager.h"
#include <random>


QLearning::QLearning(string state_EndCondition, string next_StateTable, string reward_Table)
{
	episodeEndCondition = EpisodeEndCondition::State;
	stateEndCondition = state_EndCondition;

	FileManager file;
	vector<vector<string>> next_state_table = file.GetTable(next_StateTable);
	SetNextStateTable(next_state_table);
	vector<vector<string>> reward_table = file.GetTable(reward_Table);
	SetRewardTable(reward_table);
}

QLearning::QLearning(int reward_EndCondition, string next_StateTable, string reward_Table)
{
	episodeEndCondition = EpisodeEndCondition::Reward;
	rewardEndCondition = reward_EndCondition;

	FileManager file;
	vector<vector<string>> next_state_table = file.GetTable(next_StateTable);
	SetNextStateTable(next_state_table);
	vector<vector<string>> reward_table = file.GetTable(reward_Table);
	SetRewardTable(reward_table);
}

void QLearning::Learn(string starting_State)
{
	Initialize(starting_State);
	if (episodeEndCondition == EpisodeEndCondition::State)
	{
		vector<SARS> sars;
		sarsList.push_back(sars);
		while (currentState != stateEndCondition)
		{
			Action();
		}
	}
	else if (episodeEndCondition == EpisodeEndCondition::Reward)
	{
		//ToDo
	}
}

void QLearning::Initialize(string state)
{
	for (vector<SARS>& sars : sarsList)
	{
		sars.clear();
	}
	sarsList.clear();

	currentState = state;
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
