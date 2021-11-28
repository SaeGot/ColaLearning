#include "ColaLearning.h"
#include <map>


// 신호 합치기
double Sum(Layer layer, Weight weight)
{
	double output_sum = 0.0;
	for (int n = 0; n < layer.GetNodeCount(); n++)
	{
		output_sum += layer.GetNodeValue(n) * weight.GetWeight(n);
	}
	output_sum += weight.GetWeight(layer.GetNodeCount());

	return output_sum;
}

// 신호 활성
double Activate(double value)
{
	double output_activated;
	if (value >= 1)
	{
		output_activated = 1.0;
	}
	else
	{
		output_activated = 0.0;
	}

	return output_activated;
}

// 게이트 종류
enum Gate
{
	AND,
	OR,
	NAND,
	NOR
};

int main()
{
	// 입력
	vector<vector<double>> input_list;
	input_list.push_back({ 0, 0 });
	input_list.push_back({ 0, 1 });
	input_list.push_back({ 1, 0 });
	input_list.push_back({ 1, 1 });

	//출력
	double output_sum = 0.0;
	double output_activated = 0.0;

	// 가중치
	map<Gate, Weight> weight_list;
	vector<double> weight_and = { 0.5, 0.5, 0.0 };
	vector<double> weight_or = { 1.0, 1.0, 0.0 };
	vector<double> weight_nand = { -0.5, -0.5, 1.5 };
	vector<double> weight_nor = { -1.0, -1.0, 1.0 };
	weight_list.insert({ AND, Weight(weight_and) });
	weight_list.insert({ OR, Weight(weight_or) });
	weight_list.insert({ NAND, Weight(weight_nand) });
	weight_list.insert({ NOR, Weight(weight_nor) });

	// 게이트별 테스트
	map<Gate, Weight>::iterator iter_weight;
	for (iter_weight = weight_list.begin(); iter_weight != weight_list.end(); iter_weight++)
	{
		Weight weight = iter_weight->second;
		switch (iter_weight->first)
		{
		case AND:
			printf_s("[AND 게이트] \n");
			break;
		case OR:
			printf_s("[OR 게이트] \n");
			break;
		case NAND:
			printf_s("[NAND 게이트] \n");
			break;
		case NOR:
			printf_s("[NOR 게이트] \n");
			break;
		}

		// 입력별 테스트
		vector<vector<double>>::iterator iter_input;
		for (iter_input = input_list.begin(); iter_input != input_list.end(); iter_input++)
		{
			vector<double> input = *iter_input;
			Layer layer(input);
			output_sum = Sum(layer, weight);
			output_activated = Activate(output_sum);
			printf_s("입력 : %1lf, %1lf → 출력 : %lf\n", input[0], input[1], output_activated);
		}
		printf_s("\n");
	}
}