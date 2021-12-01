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
	vector<vector<double>> weight_and = { {0.5}, {0.5}, {0.0} };
	vector<vector<double>> weight_or = { {1.0}, {1.0}, {0.0} };
	vector<vector<double>> weight_nand = { {-0.5}, {-0.5}, {1.5} };
	vector<vector<double>> weight_nor = { {-1.0}, {-1.0}, {1.0} };
	weight_list.insert({ Gate::AND, Weight(weight_and) });
	weight_list.insert({ Gate::OR, Weight(weight_or) });
	weight_list.insert({ Gate::NAND, Weight(weight_nand) });
	weight_list.insert({ Gate::NOR, Weight(weight_nor) });

	// 게이트별 테스트
	map<Gate, Weight>::iterator iter_weight;
	for (iter_weight = weight_list.begin(); iter_weight != weight_list.end(); iter_weight++)
	{
		Weight weight(iter_weight->second);
		switch (iter_weight->first)
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
		vector<vector<double>>::iterator iter_input;
		for (iter_input = input_list.begin(); iter_input != input_list.end(); iter_input++)
		{			
			vector<double> input = *iter_input;
			// 각 층 생성
			Layer layer_input(input, true);
			Layer layer_output(1);
			vector<Layer> layers( {layer_input, layer_output} );
			// 가중치 생성
			vector<Weight> weights({ weight });
			// 신경망 생성
			NeuralNetwork net(layers, weights);

			printf_s("입력 : %1lf, %1lf → 출력 : %lf\n", input[0], input[1], net.Predict()[0]);
		}
		printf_s("\n");
	}
	/*
	// XOR 게이트
	printf_s("[XOR 게이트] \n");
	vector<vector<double>>::iterator iter_input;
	for (iter_input = input_list.begin(); iter_input != input_list.end(); iter_input++)
	{
		vector<double> input = *iter_input;
		// 각 층 생성
		Layer layer_input(input, true);
		Layer layer_hidden(2);
		Layer layer_output(1);
		vector<Layer> layers({ layer_input, layer_hidden, layer_output });
		// 각 게이트 출력 계산
		vector<vector<double>> vec_weight1;
		for (int i = 0; i < 3; i++)
		{
			vec_weight1[0].push_back(weight_or[i][0]);
			vec_weight1[1].push_back(weight_nand[i][0]);
		}
		vector<Weight> weights({ weight });
		double gate_or_sum = Sum(layer1, weight_list.at(OR));
		double gate_nand_sum = Sum(layer1, weight_list.at(NAND));
		double gate_or_output = Activate(gate_or_sum);
		double gate_nand_output = Activate(gate_nand_sum);
		// 두 번째 층 입력
		vector<double> layer2_input;
		layer2_input.push_back(gate_or_output);
		layer2_input.push_back(gate_nand_output);
		Layer layer2(layer2_input);
		// 두 번째 층 출력 계산
		output_sum = Sum(layer2, weight_list.at(AND));
		output_activated = Activate(output_sum);
		printf_s("입력 : %1lf, %1lf → 출력 : %lf\n", input[0], input[1], output_activated);
	}
	printf_s("\n");	
	*/
}