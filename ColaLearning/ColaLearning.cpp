#include "ColaLearning.h"
#include <map>


// 게이트 종류
enum class Gate
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

	// 가중치
	map<Gate, Weight> weight_list;
	vector<vector<double>> weight_and = { {0.7}, {0.7}, {-1.0} };
	vector<vector<double>> weight_or = { {1.1}, {1.1}, {-1.0} };
	vector<vector<double>> weight_nand = { {-0.5}, {-0.5}, {0.7} };
	vector<vector<double>> weight_nor = { {-1.0}, {-1.0}, {0.5} };
	weight_list.insert({ Gate::AND, Weight(weight_and) });
	weight_list.insert({ Gate::OR, Weight(weight_or) });
	weight_list.insert({ Gate::NAND, Weight(weight_nand) });
	weight_list.insert({ Gate::NOR, Weight(weight_nor) });

	// 게이트별 테스트
	for (const pair<Gate, Weight> &iter_weight : weight_list)
	{
		Weight weight(iter_weight.second);
		switch (iter_weight.first)
		{
		case Gate::AND:
			printf_s("[AND 게이트] \n");
			break;
		case Gate::OR:
			printf_s("[OR 게이트] \n");
			break;
		case Gate::NAND:
			printf_s("[NAND 게이트] \n");
			break;
		case Gate::NOR:
			printf_s("[NOR 게이트] \n");
			break;
		}

		// 입력별 테스트
		for (int index = 0; index < input_list.size(); index++)
		{			
			// 각 층 생성
			Layer layer_input(input_list[index], ActivationFunction::Step, true);
			Layer layer_output(1, ActivationFunction::Step);
			vector<Layer> layers( {layer_input, layer_output} );
			// 가중치 생성
			vector<Weight> weights({ weight });
			// 신경망 생성
			NeuralNetwork net(layers, weights);

			printf_s("입력 : %1lf, %1lf → 출력 : %lf\n", input_list[index].at(0), input_list[index].at(1), net.Predict(layer_input)[0]);
		}
		printf_s("\n");
	}
	
	// XOR 게이트
	printf_s("[XOR 게이트] \n");
	vector<vector<double>>::iterator iter_input;
	for (iter_input = input_list.begin(); iter_input != input_list.end(); iter_input++)
	{
		vector<double> input = *iter_input;
		// 각 층 생성
		Layer layer_input(input, ActivationFunction::Step, true);
		Layer layer_hidden(2, ActivationFunction::Step, true);
		Layer layer_output(1, ActivationFunction::Step);
		vector<Layer> layers({ layer_input, layer_hidden, layer_output });
		// 각 가중치 생성
		vector<vector<double>> vec_weight;
		for (int i = 0; i < 3; i++)
		{
			vec_weight.push_back({ weight_or[i][0], weight_nand[i][0] });
		}
		vector<Weight> weights({ vec_weight, weight_list.at(Gate::AND) });
		// 신경망 생성
		NeuralNetwork net(layers, weights);
		printf_s("입력 : %1lf, %1lf → 출력 : %lf\n", input[0], input[1], net.Predict(layer_input)[0]);
	}
	printf_s("\n");	
	
}