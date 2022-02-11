#pragma once
#include "ReinforcementLearning.h"
#include <map>
#include <string>


class QLearning : public ReinforcementLearning
{
public:
	/**
	 * csv 파일로 상태별 행동에 따른 다음 상태 및 보상 표 생성. (행 : 행동, 열 : 상태).
	 * 상태에 따른 에피소드 종료 조건
	 *
	 * \param state_EndCondition : 상태에 따른 에피소드 종료 조건
	 * \param next_StateTable : 다음 상태 표
	 * \param reward_Table : 보상표
	 */
	QLearning(string state_EndCondition, string next_StateTable, string reward_Table);
	/**
	 * csv 파일로 상태별 행동에 따른 다음 상태 및 보상 표 생성. (행 : 행동, 열 : 상태)
	 * 보상에 따른 에피소드 종료 조건
	 * 
	 * \param min_RewardEndCondition : 보상에 따른 에피소드 종료 조건 (0 이하의 최소값, 0일 경우 이 조건은 무시)
	 * \param max_RewardEndCondition : 보상에 따른 에피소드 종료 조건 (0 초과의 최대값)
	 * \param next_StateTable : 다음 상태 표
	 * \param reward_Table : 보상표
	 */
	QLearning(double min_RewardEndCondition, double max_RewardEndCondition, string next_StateTable, string reward_Table);
	/**
	 * 모든 에피소드 초기화.
	 * 
	 */
	void Initialize();
	/**
	 * 특정 상태에서 시작하여 하나의 에피소드를 진행 및 학습.
	 * 
	 * \param starting_State : 시작 상태
	 * \param discount_Factor : 감가율
	 * \param epsilon_Greedy
	 */
	void Learn(string starting_State, double discount_Factor, EpsilonGreedy& epsilon_Greedy);
	/**
	 * 행동 진행하여 다음 상태로 전이.
	 * 
	 * \param action : 행동
	 */
	void Action(string action = "");
	/**
	 * 시작 상태에서 시작하여 가장 좋은 행동들을 가져오기.
	 * 
	 * \param starting_State : 시작 상태
	 * \return 최적 행동들
	 */
	vector<string> GetBest(string starting_State);

private:
	struct StateAction
	{
		string state;
		string action;

		bool operator<(const StateAction& rhs) const;
	};
	// 각 상태별 행동에 대해 어떤 상태가 되는지에 대한 표
	map<StateAction, string> nextStateTable;
	// 각 상태별 행동에 대한 보상 표
	map<StateAction, double> rewardTable;
	// 각 상태별 가능한 행동
	map<string, vector<string>> enableAction;
	// 각 상태별 Q값 표
	map<StateAction, double> QTable;

	/**
	 * nextStateTable 설정.
	 * 
	 * \param table : nextStateTable
	 */
	void SetNextStateTable(const vector<vector<string>>& table);
	/**
	 * rewardTable 설정.
	 * 
	 * \param table : rewardTable
	 */
	void SetRewardTable(const vector<vector<string>>& table);
	/**
	 * QTable 업데이트.
	 * 
	 * \param discount_Factor : 감가율
	 */
	void UpdateQTable(double discount_Factor);
	/**
	 * 현재 상태에서 가장 좋은 행동 가져오기.
	 * 
	 * \param state : 현재 상태
	 * \return 최적 행동
	 */
	string GetBestAction(string state);
	/**
	 * 랜덤선택 여부 가져오기.
	 * 
	 * \param epsilon_Greedy : Epsilon Greedy
	 * \param step : 스탭
	 * \return 랜덤선택 여부
	 */
	bool GetRandomPolicy(EpsilonGreedy& epsilon_Greedy, size_t step);
	/**
	 * 보상 종료 조건 여부 가져오기.
	 * 
	 * \param cumulative_Reward : 누적 보상
	 * \return 보상 종료 조건 여부
	 */
	bool CheckRewardEndCondition(double cumulative_Reward);
};
