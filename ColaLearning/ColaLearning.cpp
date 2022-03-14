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

void GateTest()
{	// 입력
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
	for (const pair<Gate, Weight>& iter_weight : weight_list)
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
			Layer layer_input(input_list[index], Layer::ActivationFunction::Step, true);
			Layer layer_output(1, Layer::ActivationFunction::Step, false);
			Layer layers[2] = { layer_input, layer_output };
			// 가중치 생성
			Weight weights[1] = { weight };
			// 신경망 생성
			NeuralNetwork net(layers, 2, weights);

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
		Layer layer_input(input, Layer::ActivationFunction::Step, true);
		Layer layer_hidden(2, Layer::ActivationFunction::Step, true);
		Layer layer_output(1, Layer::ActivationFunction::Step, false);
		Layer* layers = new Layer[3]{ layer_input, layer_hidden, layer_output };
		// 각 가중치 생성
		vector<vector<double>> vec_weight;
		for (int i = 0; i < 3; i++)
		{
			vec_weight.push_back({ weight_or[i][0], weight_nand[i][0] });
		}
		Weight* weights = new Weight[2] { vec_weight, weight_list.at(Gate::AND) };
		// 신경망 생성
		NeuralNetwork net(layers, 3, weights);
		printf_s("입력 : %1lf, %1lf → 출력 : %lf\n", input[0], input[1], net.Predict(layer_input)[0]);
	}
	printf_s("\n");
}

void LearnTest()
{
	FileManager file = FileManager("NeuralNetwork_Example.csv");
	Layer layer_input(1);
	Layer layer_hidden(3);
	Layer layer_output(1);
	Layer layers[3] = { layer_input, layer_hidden, layer_output };
	int layer_count = sizeof(layers) / sizeof(Layer);
	NeuralNetwork net(layers, layer_count);
	double input = file.GetData(0, 0);

	vector<double> data = {input};
	Layer layer_inputdata(data);
	double output = net.Predict(layer_inputdata)[0];
	printf("%lf\n", output);

	vector<Layer> input_learning_layers;
	vector<Layer> targe_layers;
	int row_count = file.GetRowCount();
	for (int n = 0; n < row_count; n++)
	{
		vector<double> input_value = { file.GetData(n, 0) };
		Layer input_learning_layer(input_value);
		vector<double> output_value = { file.GetData(n, 1) };
		Layer targe_layer(output_value);
		input_learning_layers.push_back(input_learning_layer);
		targe_layers.push_back(targe_layer);
	}
	Optimizer* optimizer = new GradientDescent(0.005);
	for (int n = 0; n < 1000; n++)
	{
		net.Learn(input_learning_layers, targe_layers, optimizer, NeuralNetwork::ErrorType::SquareError);
		output = net.Predict(layer_inputdata)[0];
		if ((n + 1) % 100 == 0)
		{
			printf("learn %d, target = 7, predict = %lf\n", n + 1, output);
		}
	}
	output = net.Predict(input_learning_layers[1])[0];
	printf("target = 10, predict = %lf\n", output);
	output = net.Predict(input_learning_layers[2])[0];
	printf("target = 31, predict =%lf\n", output);
	output = net.Predict(input_learning_layers[3])[0];
	printf("target = 61, predict =%lf\n", output);
}

void QLearningTestStateEnd()
{
	QLearning q_learning("55", "QLearning_Example_nextStateTable.csv", "QLearning_Example_rewardTable.csv");
	QLearning::EpsilonGreedy epsilon;
	epsilon.beginningValue = 1;
	epsilon.interval = 2;
	epsilon.gamma = 0.9999;

	for (int n = 0; n < 100; n++)
	{
		q_learning.Learn("00", 0.8, epsilon);
	}
	vector<string> best_way = q_learning.GetBest("00");
	for (string state : best_way)
	{
		printf("%s\n", state.c_str());
	}
}

void QLearningTestRewardEnd()
{
	QLearning q_learning(0, 200, "QLearning_Example_nextStateTable.csv", "QLearning_Example_rewardTable2.csv");
	QLearning::EpsilonGreedy epsilon;
	epsilon.beginningValue = 1;
	epsilon.interval = 2;
	epsilon.gamma = 0.9999;

	for (int n = 0; n < 100; n++)
	{
		q_learning.Learn("00", 0.8, epsilon);
	}
	vector<string> best_way = q_learning.GetBest("00");
	for (string state : best_way)
	{
		printf("%s\n", state.c_str());
	}
}

void OneHotEncodingTest()
{
	vector<FileManager::Type> types = { FileManager::Type::String, FileManager::Type::String, FileManager::Type::Real };
	FileManager file = FileManager("OneHotEncoding_Example.csv", types, FileManager::Type::OneHot);
	vector<double> input_1_encoding = file.GetEncodingData(0, 0);
	vector<double> input_2_encoding = file.GetEncodingData(0, 1);

	Layer layer_input(input_1_encoding.size() + input_2_encoding.size());
	Layer layer_output(1);
	Layer layers[2] = { layer_input, layer_output };
	int layer_count = sizeof(layers) / sizeof(Layer);
	NeuralNetwork net(layers, layer_count);

	vector<double> data = file.GetEncodingData(0, {0, 1});
	Layer layer_inputdata(data);
	double output = net.Predict(layer_inputdata)[0];
	printf("%lf\n", output);

	vector<Layer> input_learning_layers;
	vector<Layer> targe_layers;
	int row_count = file.GetRowCount();
	for (int n = 0; n < row_count; n++)
	{
		vector<double> input_value = file.GetEncodingData(n, {0, 1});
		Layer input_learning_layer(input_value);
		vector<double> output_value = { file.GetData(n, 2) };
		Layer targe_layer(output_value);
		input_learning_layers.push_back(input_learning_layer);
		targe_layers.push_back(targe_layer);
	}
	Optimizer* optimizer = new GradientDescent(0.005);
	for (int n = 0; n < 1000; n++)
	{
		net.Learn(input_learning_layers, targe_layers, optimizer, NeuralNetwork::ErrorType::SquareError);
		output = net.Predict(layer_inputdata)[0];
		if ((n + 1) % 100 == 0)
		{
			printf("learn %d, target = 2, predict = %lf\n", n+1, output);
		}
	}
	output = net.Predict(input_learning_layers[1])[0];
	printf("target = 3, predict = %lf\n", output);
	output = net.Predict(input_learning_layers[2])[0];
	printf("target = 4, predict =%lf\n", output);
	output = net.Predict(input_learning_layers[3])[0];
	printf("target = 5, predict =%lf\n", output);
	printf("\n");
}

void CrossEntropyTest()
{
	FileManager file = FileManager("OneHotEncoding_Example.csv", FileManager::Type::String, FileManager::Type::OneHot);

	vector<double> input_1_encoding = file.GetEncodingData(0, 0);
	vector<double> input_2_encoding = file.GetEncodingData(0, 1);
	vector<double> output_encoding = file.GetEncodingData(0, 2);
	Layer layer_input(input_1_encoding.size() + input_2_encoding.size());
	Layer layer_output(output_encoding.size(), Layer::ActivationFunction::Softmax);
	Layer layers[2] = { layer_input, layer_output };
	int layer_count = sizeof(layers) / sizeof(Layer);
	NeuralNetwork net(layers, layer_count);
	vector<double> data = file.GetEncodingData(0, { 0, 1 });
	Layer layer_inputdata(data);

	vector<Layer> input_learning_layers;
	vector<Layer> targe_layers;
	int row_count = file.GetRowCount();
	for (int n = 0; n < row_count; n++)
	{
		vector<double> input_value = file.GetEncodingData(n, { 0, 1 });
		Layer input_learning_layer(input_value);
		vector<double> output_value = { file.GetEncodingData(n, 2) };
		Layer targe_layer(output_value, Layer::ActivationFunction::Softmax);
		input_learning_layers.push_back(input_learning_layer);
		targe_layers.push_back(targe_layer);
	}
	Optimizer* optimizer = new GradientDescent(0.05);
	vector<double> output;
	for (int n = 0; n < 1000; n++)
	{
		net.Learn(input_learning_layers, targe_layers, optimizer, NeuralNetwork::ErrorType::CrossEntropy);
		output = net.Predict(layer_inputdata);
		if ((n + 1) % 100 == 0)
		{
			printf("learn %d, target = 2, predict = %lf, %lf, %lf, %lf\n", n + 1, output[0], output[1], output[2], output[3]);
			printf("Sum = %lf\n", output[0] + output[1] + output[2] + output[3]);
		}
	}
	output = net.Predict(input_learning_layers[1]);
	printf("target = 3, predict = %lf, %lf, %lf, %lf\n", output[0], output[1], output[2], output[3]);
	printf("Sum = %lf\n", output[0] + output[1] + output[2] + output[3]);
	output = net.Predict(input_learning_layers[2]);
	printf("target = 4, predict = %lf, %lf, %lf, %lf\n", output[0], output[1], output[2], output[3]);
	printf("Sum = %lf\n", output[0] + output[1] + output[2] + output[3]);
	output = net.Predict(input_learning_layers[3]);
	printf("target = 5, predict = %lf, %lf, %lf, %lf\n", output[0], output[1], output[2], output[3]);
	printf("Sum = %lf\n", output[0] + output[1] + output[2] + output[3]);
	printf("\n");
}

int main()
{
	GateTest();
	LearnTest();
	printf("State End\n");
	QLearningTestStateEnd();
	printf("Reward End\n");
	QLearningTestRewardEnd();
	printf("QLearning End\n");
	OneHotEncodingTest();
	CrossEntropyTest();
}