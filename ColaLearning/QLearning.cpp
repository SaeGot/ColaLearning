#include "QLearning.h"
#include "FileManager.h"


QLearning::QLearning(string file_Name)
{
	FileManager file;
	vector<vector<string>> table = file.GetTable(file_Name);

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
		for (int j = 1; j < row.size(); j++)
		{
			StateAction state_action = { first_row[j], row[0] };
			nextStateTable.insert({ state_action, row[j] });
		}
	}
}

void QLearning::Learn(string state)
{
	Initialize(state);
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
		//ToDo ·£´ý¼±ÅÃ		
	}
	else
	{
		StateAction state_action = { current_state, action };
		next_state = nextStateTable[state_action];
	}
	stateList.back().push_back(next_state);
}

bool QLearning::StateAction::operator<(const StateAction& rhs) const
{
	if (state != rhs.state)
	{
		return state < rhs.state;
	}
	
	return action < rhs.action;
}
