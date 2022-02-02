#include "QLearning.h"
#include "FileManager.h"
#include <random>


QLearning::QLearning(string state_EndCondition, string file_Name)
{
	episodeEndCondition = EpisodeEndCondition::State;
	stateEndCondition = state_EndCondition;

	FileManager file;
	vector<vector<string>> table = file.GetTable(file_Name);

	SetTable(table);
}

QLearning::QLearning(int reward_EndCondition, string file_Name)
{
	episodeEndCondition = EpisodeEndCondition::Reward;
	rewardEndCondition = reward_EndCondition;

	FileManager file;
	vector<vector<string>> table = file.GetTable(file_Name);

	SetTable(table);
}

void QLearning::Learn(string state)
{
	Initialize(state);
	if (episodeEndCondition == EpisodeEndCondition::State)
	{
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
	vector<string> current_episode_states = { state };
	stateList.push_back(current_episode_states);
}

void QLearning::Action(string action)
{
	vector<string> current_episode_states = stateList.back();
	string current_state = current_episode_states.back();
	string next_state;
	if (action == "")
	{
		const vector<string>& enable_action = enableAction[current_state];

		random_device rd;
		mt19937_64 gen(rd());
		int action_count = static_cast<int>(enable_action.size() - 1);
		uniform_int_distribution<int> random_number(0, action_count);

		StateAction state_action = { current_state, enable_action[random_number(gen)] };
		next_state = nextStateTable[state_action];
	}
	else
	{
		StateAction state_action = { current_state, action };
		next_state = nextStateTable[state_action];
	}
	stateList.back().push_back(next_state);
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

void QLearning::SetTable(const vector<vector<string>>& table)
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
