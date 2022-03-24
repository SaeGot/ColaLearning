#include "ColaLearning.h"
#include <map>
#include <filesystem>
#include <opencv2/opencv.hpp>
using namespace cv;


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
			Layer layer_input(input_list[index], Layer::LayerType::FullyConnected, Layer::ActivationFunction::Step, true);
			Layer layer_output(1, Layer::LayerType::FullyConnected, Layer::ActivationFunction::Step, false);
			vector<Layer*> layers = { new Layer(layer_input), new Layer(layer_output) };
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
		Layer layer_input(input, Layer::LayerType::FullyConnected, Layer::ActivationFunction::Step, true);
		Layer layer_hidden(2, Layer::LayerType::FullyConnected, Layer::ActivationFunction::Step, true);
		Layer layer_output(1, Layer::LayerType::FullyConnected, Layer::ActivationFunction::Step, false);
		vector<Layer*> layers = { new Layer(layer_input), new Layer(layer_hidden), new Layer(layer_output) };
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
	vector<Layer*> layers = { new Layer(layer_input), new Layer(layer_hidden), new Layer(layer_output) };
	//int layer_count = sizeof(layers) / sizeof(Layer);
	NeuralNetwork net(layers, layers.size());
	double input = file.GetData(0, 0);

	vector<double> data = {input};
	Layer layer_inputdata(data);
	double output = net.Predict(layer_inputdata)[0];
	printf("%lf\n", output);

	vector<Layer*> input_learning_layers;
	vector<Layer*> targe_layers;
	int row_count = file.GetRowCount();
	for (int n = 0; n < row_count; n++)
	{
		vector<double> input_value = { file.GetData(n, 0) };
		Layer input_learning_layer(input_value);
		vector<double> output_value = { file.GetData(n, 1) };
		Layer target_layer(output_value);
		input_learning_layers.push_back(new Layer(input_learning_layer));
		targe_layers.push_back(new Layer(target_layer));
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
	output = net.Predict(*input_learning_layers[1])[0];
	printf("target = 10, predict = %lf\n", output);
	output = net.Predict(*input_learning_layers[2])[0];
	printf("target = 31, predict =%lf\n", output);
	output = net.Predict(*input_learning_layers[3])[0];
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
	vector<Layer*> layers = { new Layer(layer_input), new Layer(layer_output) };
	//int layer_count = sizeof(layers) / sizeof(Layer);
	NeuralNetwork net(layers, layers.size());

	vector<double> data = file.GetEncodingData(0, {0, 1});
	Layer layer_inputdata(data);
	double output = net.Predict(layer_inputdata)[0];
	printf("%lf\n", output);

	vector<Layer*> input_learning_layers;
	vector<Layer*> targe_layers;
	int row_count = file.GetRowCount();
	for (int n = 0; n < row_count; n++)
	{
		vector<double> input_value = file.GetEncodingData(n, {0, 1});
		Layer input_learning_layer(input_value);
		vector<double> output_value = { file.GetData(n, 2) };
		Layer target_layer(output_value);
		input_learning_layers.push_back(new Layer(input_learning_layer));
		targe_layers.push_back(new Layer(target_layer));
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
	output = net.Predict(*input_learning_layers[1])[0];
	printf("target = 3, predict = %lf\n", output);
	output = net.Predict(*input_learning_layers[2])[0];
	printf("target = 4, predict =%lf\n", output);
	output = net.Predict(*input_learning_layers[3])[0];
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
	Layer layer_output(output_encoding.size(), Layer::LayerType::FullyConnected, Layer::ActivationFunction::Softmax);
	vector<Layer*> layers = { new Layer(layer_input), new Layer(layer_output) };
	//int layer_count = sizeof(layers) / sizeof(Layer);
	NeuralNetwork net(layers, layers.size());
	vector<double> data = file.GetEncodingData(0, { 0, 1 });
	Layer layer_inputdata(data);

	vector<Layer*> input_learning_layers;
	vector<Layer*> targe_layers;
	int row_count = file.GetRowCount();
	for (int n = 0; n < row_count; n++)
	{
		vector<double> input_value = file.GetEncodingData(n, { 0, 1 });
		Layer input_learning_layer(input_value);
		vector<double> output_value = { file.GetEncodingData(n, 2) };
		Layer target_layer(output_value, Layer::LayerType::FullyConnected, Layer::ActivationFunction::Softmax);
		input_learning_layers.push_back(new Layer(input_learning_layer));
		targe_layers.push_back(new Layer(target_layer));
	}
	Optimizer* optimizer = new GradientDescent(0.01);
	map<Tensor, double> output;
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
	output = net.Predict(*input_learning_layers[1]);
	printf("target = 3, predict = %lf, %lf, %lf, %lf\n", output[0], output[1], output[2], output[3]);
	printf("Sum = %lf\n", output[0] + output[1] + output[2] + output[3]);
	output = net.Predict(*input_learning_layers[2]);
	printf("target = 4, predict = %lf, %lf, %lf, %lf\n", output[0], output[1], output[2], output[3]);
	printf("Sum = %lf\n", output[0] + output[1] + output[2] + output[3]);
	output = net.Predict(*input_learning_layers[3]);
	printf("target = 5, predict = %lf, %lf, %lf, %lf\n", output[0], output[1], output[2], output[3]);
	printf("Sum = %lf\n", output[0] + output[1] + output[2] + output[3]);
	printf("\n");
}

void Layer2DTest()
{
	Layer layer_input(3, 3);
	Layer layer_output(3, 3);
	vector<Layer*> layers = { new Layer(layer_input), new Layer(layer_output) };
	//int layer_count = sizeof(layers) / sizeof(Layer);
	NeuralNetwork net(layers, layers.size());
	vector<Layer*> input_learning_layers;
	vector<Layer*> targe_layers;
	for (int n = 0; n < 5; n++)
	{
		map<Tensor, double> input_data;
		map<Tensor, double> output_data;
		for (int x = 0; x < 3; x++)
		{
			for (int y = 0; y < 3; y++)
			{
				Tensor tensor = Tensor(x, y);
				input_data.emplace(tensor, 10 * x + y + (n * 100));
				output_data.emplace(tensor, 10 * x + y + (n * 1000));
			}
		}
		FullyConnectedLayer input_learning_layer(input_data);
		FullyConnectedLayer target_layer(output_data);
		input_learning_layers.push_back(new Layer(input_learning_layer));
		targe_layers.push_back(new Layer(target_layer));
	}
	Optimizer* optimizer = new GradientDescent(0.01);
	map<Tensor, double> output;
	for (int n = 0; n < 1000; n++)
	{
		net.Learn(input_learning_layers, targe_layers, optimizer, NeuralNetwork::ErrorType::CrossEntropy);
		output = net.Predict(*input_learning_layers[0]);
	}
	output = net.Predict(*input_learning_layers[0]);
	output = net.Predict(*input_learning_layers[1]);
	output = net.Predict(*input_learning_layers[2]);
}

void ConvTest()
{
	Layer layer_input(3, 3);
	ConvolutionLayer layer_hid(2, Tensor(3, 3), Layer::ActivationFunction::ReLU, true, Tensor(1, 1), Tensor(1, 1));
	PoolingLayer layer_pool(Tensor(2, 2), Tensor(2, 2));
	Layer layer_output(1);
	vector<Layer*> layers = { new Layer(layer_input), new ConvolutionLayer(layer_hid), new PoolingLayer(layer_pool), new Layer(layer_output) };

	NeuralNetwork net(layers, layers.size());
	vector<Layer*> input_learning_layers;
	vector<Layer*> targe_layers;
	for (int n = 0; n < 5; n++)
	{
		map<Tensor, double> input_data;
		map<Tensor, double> output_data;
		for (int x = 0; x < 3; x++)
		{
			for (int y = 0; y < 3; y++)
			{
				Tensor tensor = Tensor(x, y);
				input_data.emplace(tensor, 10 * x + y + (n * 100));
			}
		}
		FullyConnectedLayer input_learning_layer(input_data);
		vector<double> target_data = { (double)n };
		FullyConnectedLayer target_layer(target_data);
		input_learning_layers.push_back(new Layer(input_learning_layer));
		targe_layers.push_back(new Layer(target_layer));
	}
	Optimizer* optimizer = new GradientDescent(0.01);
	map<Tensor, double> output;
	output = net.Predict(*input_learning_layers[0]);
	for (int n = 0; n < 1000; n++)
	{
		net.Learn(input_learning_layers, targe_layers, optimizer, NeuralNetwork::ErrorType::SquareError);
		output = net.Predict(*input_learning_layers[1]);
	}
	output = net.Predict(*input_learning_layers[0]);
	output = net.Predict(*input_learning_layers[1]);
	output = net.Predict(*input_learning_layers[2]);
}

void MnistTest()
{
	//namedWindow("image");
	int test_numbers = 3;
	vector<Layer*> input_learning_layers;
	vector<Layer*> target_learning_layers;
	string training_path = "mnist_png/training/";
	int interval = 0;
	for (const filesystem::directory_entry folder : filesystem::directory_iterator(training_path))
	{
		string folder_str = folder.path().string();
		char* folder_char = new char[folder_str.size() + 1];
		strcpy_s(folder_char, folder_str.size() + 1, folder_str.c_str());
		char* dummy = NULL;
		char* number_char = strtok_s(folder_char, training_path.c_str(), &dummy);
		string number_str(number_char);
		int number_int = stoi(number_str);
		if (number_int < test_numbers)
		{
			for (const filesystem::directory_entry file : filesystem::directory_iterator(folder))
			{
				if (interval % 2000 == 0)
				{
					string file_str = file.path().string();
					//string file_name = file.path().relative_path().string();
					Mat image = imread(file_str, IMREAD_GRAYSCALE);
					Tensor xy = Tensor(image.rows, image.cols, 1);
					map<Tensor, double> data;
					for (Tensor tensor : xy.GetTensors())
					{
						vector<int> xychannel = tensor.GetXYChannel();
						data.emplace(tensor, image.at<uchar>(xychannel[0], xychannel[1]));
					}
					FullyConnectedLayer input_layer(data);
					input_learning_layers.push_back(new Layer(input_layer));
					vector<double> target_data;
					for (int n = 0; n < test_numbers; n++)
					{
						if (number_int == n)
						{
							target_data.push_back(1);
						}
						else
						{
							target_data.push_back(0);
						}
					}
					FullyConnectedLayer target_layers(target_data);
					target_learning_layers.push_back(new Layer(target_layers));
				}
				interval++;
			}
		}		
	}
	
	vector<Layer*> input_test_layers;
	vector<Layer*> target_test_layers;
	string testing_path = "mnist_png/testing/";
	for (const filesystem::directory_entry folder : filesystem::directory_iterator(testing_path))
	{
		string folder_str = folder.path().string();
		char* folder_char = new char[folder_str.size() + 1];
		strcpy_s(folder_char, folder_str.size() + 1, folder_str.c_str());
		char* dummy = NULL;
		char* number_char = strtok_s(folder_char, testing_path.c_str(), &dummy);
		string number(number_char);
		string number_str(number_char);
		int number_int = stoi(number_str);
		if (number_int < test_numbers)
		{
			for (const filesystem::directory_entry file : filesystem::directory_iterator(folder))
			{
				if (interval % 1000 == 0)
				{
					string file_str = file.path().string();
					//string file_name = file.path().relative_path().string();
					Mat image = imread(file_str);
					Tensor xy = Tensor(image.rows, image.cols, 1);
					map<Tensor, double> data;
					for (Tensor tensor : xy.GetTensors())
					{
						vector<int> xychannel = tensor.GetXYChannel();
						data.emplace(tensor, image.at<uchar>(xychannel[0], xychannel[1]));
					}
					FullyConnectedLayer input_layer(data);
					input_test_layers.push_back(new Layer(input_layer));
					vector<double> target_data;
					for (int n = 0; n < test_numbers; n++)
					{
						if (number_int == n)
						{
							target_data.push_back(1);
						}
						else
						{
							target_data.push_back(0);
						}
					}
					FullyConnectedLayer target_layers(target_data);
					target_test_layers.push_back(new Layer(target_layers));
				}
				interval++;
			}
		}
		
	}


	Layer layer_input(28, 28);
	ConvolutionLayer layer_hid1(5, Tensor(5, 5), Layer::ActivationFunction::Tanh, true, Tensor(2, 2));
	ConvolutionLayer layer_hid2(2, Tensor(5, 5), Layer::ActivationFunction::Tanh, true);
	FullyConnectedLayer layer_hid3(32, Layer::ActivationFunction::ReLU, true);
	Layer layer_output(test_numbers, Layer::LayerType::FullyConnected, Layer::ActivationFunction::Softmax);
	vector<Layer*> layers = { new Layer(layer_input), new ConvolutionLayer(layer_hid1), new FullyConnectedLayer(layer_hid3),new Layer(layer_output) };

	NeuralNetwork net(layers, layers.size());

	Optimizer* optimizer = new GradientDescent(0.005);
	map<Tensor, double> output;
	for (int epoch = 0; epoch < 1000; epoch++)
	{
		net.Learn(input_learning_layers, target_learning_layers, optimizer, NeuralNetwork::ErrorType::CrossEntropy);

		for (int n = 0; n < input_test_layers.size(); n++)
		{
			output = net.Predict(*input_test_layers[n]);
			double target_value = 0;
			double predict = 0;
			double max_percentage = 0;
			vector<double> output_percent;
			for (const pair<Tensor, double>& tensor : output)
			{
				double current_value = target_test_layers[n]->GetNodeValue(tensor.first);
				if (current_value == 1)
				{
					target_value = tensor.first.GetXYChannel()[0];
				}
				double percent = tensor.second;
				if (tensor.second > max_percentage)
				{
					predict = tensor.first.GetXYChannel()[0];
					max_percentage = tensor.second;
				}
				output_percent.push_back(tensor.second);
			}
			printf("step = %d : ", epoch);
			printf("target = %lf, predict = %lf, %lf, %lf\n", target_value, output_percent[0], output_percent[1], output_percent[2]);
		}
	}
	for (int n = 0; n < input_test_layers.size(); n++)
	{
		output = net.Predict(*input_test_layers[n]);
		double target_value = 0;
		double predict = 0;
		double max_percentage = 0;
		for (const pair<Tensor, double>& tensor : output)
		{
			double current_value = target_test_layers[n]->GetNodeValue(tensor.first);
			if (current_value == 1)
			{
				target_value = tensor.first.GetXYChannel()[0];
			}
			double percent = tensor.second;
			if (tensor.second > max_percentage)
			{
				predict = tensor.first.GetXYChannel()[0];
				max_percentage = tensor.second;
			}
		}
		printf("target = %lf, predict = %lf \n", target_value, predict);
	}
	//double data = img.at<uchar>();
	//imshow("image", img);

	//waitKey();
}

void CrossTest()
{
	//namedWindow("image");
	int test_numbers = 2;
	vector<Layer*> input_learning_layers;
	vector<Layer*> target_learning_layers;
	string training_path = "cross/training/";
	int interval = 0;
	for (const filesystem::directory_entry folder : filesystem::directory_iterator(training_path))
	{
		string folder_str = folder.path().string();
		char* folder_char = new char[folder_str.size() + 1];
		strcpy_s(folder_char, folder_str.size() + 1, folder_str.c_str());
		char* dummy = NULL;
		char* number_char = strtok_s(folder_char, training_path.c_str(), &dummy);
		string number_str(number_char);
		int number_int = stoi(number_str);
		if (number_int < test_numbers)
		{
			for (const filesystem::directory_entry file : filesystem::directory_iterator(folder))
			{
				if (interval % 1 == 0)
				{
					string file_str = file.path().string();
					//string file_name = file.path().relative_path().string();
					Mat image = imread(file_str, IMREAD_GRAYSCALE);
					Tensor xy = Tensor(image.rows, image.cols, 1);
					double aa = image.at<uchar>(14, 14);

					map<Tensor, double> data;
					for (Tensor tensor : xy.GetTensors())
					{
						vector<int> xychannel = tensor.GetXYChannel();
						data.emplace(tensor, image.at<uchar>(xychannel[0], xychannel[1]));
					}
					FullyConnectedLayer input_layer(data);
					input_learning_layers.push_back(new Layer(input_layer));
					vector<double> target_data;
					for (int n = 0; n < test_numbers; n++)
					{
						if (number_int == n)
						{
							target_data.push_back(1);
						}
						else
						{
							target_data.push_back(0);
						}
					}
					FullyConnectedLayer target_layers(target_data);
					target_learning_layers.push_back(new Layer(target_layers));
				}
				interval++;
			}
		}
	}
	/*
	vector<Layer*> input_test_layers;
	vector<Layer*> target_test_layers;
	string testing_path = "cross/testing/";
	for (const filesystem::directory_entry folder : filesystem::directory_iterator(testing_path))
	{
		string folder_str = folder.path().string();
		char* folder_char = new char[folder_str.size() + 1];
		strcpy_s(folder_char, folder_str.size() + 1, folder_str.c_str());
		char* dummy = NULL;
		char* number_char = strtok_s(folder_char, testing_path.c_str(), &dummy);
		string number(number_char);
		string number_str(number_char);
		int number_int = stoi(number_str);
		if (number_int < test_numbers)
		{
			for (const filesystem::directory_entry file : filesystem::directory_iterator(folder))
			{
				if (interval % 1 == 0)
				{
					string file_str = file.path().string();
					//string file_name = file.path().relative_path().string();
					Mat image = imread(file_str);
					Tensor xy = Tensor(image.rows, image.cols, 1);
					map<Tensor, double> data;
					for (Tensor tensor : xy.GetTensors())
					{
						vector<int> xychannel = tensor.GetXYChannel();
						data.emplace(tensor, image.at<uchar>(xychannel[0], xychannel[1]));
					}
					FullyConnectedLayer input_layer(data);
					input_test_layers.push_back(new Layer(input_layer));
					vector<double> target_data;
					for (int n = 0; n < test_numbers; n++)
					{
						if (number_int == n)
						{
							target_data.push_back(1);
						}
						else
						{
							target_data.push_back(0);
						}
					}
					FullyConnectedLayer target_layers(target_data);
					target_test_layers.push_back(new Layer(target_layers));
				}
				interval++;
			}
		}

	}

	*/
	Layer layer_input(28, 28);
	ConvolutionLayer layer_hid1(1, Tensor(5, 5), Layer::ActivationFunction::Tanh, true, Tensor(2, 2));
	PoolingLayer layer_pooling1(Tensor(2, 2), Tensor(2, 2));
	ConvolutionLayer layer_hid2(2, Tensor(5, 5), Layer::ActivationFunction::Tanh, true);
	FullyConnectedLayer layer_hid3(32, Layer::ActivationFunction::ReLU, true);
	Layer layer_output(test_numbers, Layer::LayerType::FullyConnected, Layer::ActivationFunction::Softmax);
	vector<Layer*> layers = { new Layer(layer_input), new ConvolutionLayer(layer_hid1), new PoolingLayer(layer_pooling1), new Layer(layer_output) };

	NeuralNetwork net(layers, layers.size());

	Optimizer* optimizer = new GradientDescent(0.01);
	map<Tensor, double> output;
	output = net.Predict(*input_learning_layers[0]);
	printf("0predict = %lf", output[Tensor(0)]);
	output = net.Predict(*input_learning_layers[1]);
	printf("1predict = %lf\n", output[Tensor(0)]);
	for (int n = 0; n < 1000; n++)
	{
		net.Learn(input_learning_layers, target_learning_layers, optimizer, NeuralNetwork::ErrorType::CrossEntropy);
		output = net.Predict(*input_learning_layers[0]);
		printf("0predict = %lf, %lf", output[Tensor(0)], output[Tensor(1)]);
		output = net.Predict(*input_learning_layers[1]);
		printf("1predict = %lf, %lf\n", output[Tensor(0)], output[Tensor(1)]);
	}
	for (int n = 0; n < input_learning_layers.size(); n++)
	{
		output = net.Predict(*input_learning_layers[n]);
		double target_value = 0;
		double predict = 0;
		double max_percentage = 0;
		vector<double> predict_percent;
		for (const pair<Tensor, double>& tensor : output)
		{
			double current_value = target_learning_layers[n]->GetNodeValue(tensor.first);
			if (current_value == 1)
			{
				target_value = tensor.first.GetXYChannel()[0];
			}
			double percent = tensor.second;
			if (tensor.second > max_percentage)
			{
				predict = tensor.first.GetXYChannel()[0];
				max_percentage = tensor.second;
			}

			predict_percent.push_back(tensor.second);
		}
		printf("target = %lf, predict = %lf, %lf \n", target_value, predict_percent[0], predict_percent[1]);
	}
}

void MiniCrossTest()
{
	//namedWindow("image");
	int test_numbers = 2;
	vector<Layer*> input_learning_layers;
	vector<Layer*> target_learning_layers;
	string training_path = "minicross/training/";
	int interval = 0;
	for (const filesystem::directory_entry folder : filesystem::directory_iterator(training_path))
	{
		string folder_str = folder.path().string();
		char* folder_char = new char[folder_str.size() + 1];
		strcpy_s(folder_char, folder_str.size() + 1, folder_str.c_str());
		char* dummy = NULL;
		char* number_char = strtok_s(folder_char, training_path.c_str(), &dummy);
		string number_str(number_char);
		int number_int = stoi(number_str);
		if (number_int < test_numbers)
		{
			for (const filesystem::directory_entry file : filesystem::directory_iterator(folder))
			{
				if (interval % 1 == 0)
				{
					string file_str = file.path().string();
					//string file_name = file.path().relative_path().string();
					Mat image = imread(file_str, IMREAD_GRAYSCALE);
					Tensor xy = Tensor(image.rows, image.cols, 1);

					map<Tensor, double> data;
					for (Tensor tensor : xy.GetTensors())
					{
						vector<int> xychannel = tensor.GetXYChannel();
						data.emplace(tensor, image.at<uchar>(xychannel[0], xychannel[1]));
					}
					FullyConnectedLayer input_layer(data);
					input_learning_layers.push_back(new Layer(input_layer));
					vector<double> target_data;
					for (int n = 0; n < test_numbers; n++)
					{
						if (number_int == n)
						{
							target_data.push_back(1);
						}
						else
						{
							target_data.push_back(0);
						}
					}
					FullyConnectedLayer target_layers(target_data);
					target_learning_layers.push_back(new Layer(target_layers));
				}
				interval++;
			}
		}
	}


	Layer layer_input(2, 2);
	ConvolutionLayer layer_hid1(1, Tensor(2, 2), Layer::ActivationFunction::Tanh, true, Tensor(1, 1));
	PoolingLayer layer_pooling1(Tensor(2, 2), Tensor(2, 2));
	ConvolutionLayer layer_hid2(2, Tensor(5, 5), Layer::ActivationFunction::Tanh, true);
	FullyConnectedLayer layer_hid3(32, Layer::ActivationFunction::ReLU, true);
	Layer layer_output(test_numbers, Layer::LayerType::FullyConnected, Layer::ActivationFunction::Softmax);
	vector<Layer*> layers = { new Layer(layer_input), new ConvolutionLayer(layer_hid1), new Layer(layer_output) };

	NeuralNetwork net(layers, layers.size());

	Optimizer* optimizer = new GradientDescent(0.001);
	map<Tensor, double> output;
	output = net.Predict(*input_learning_layers[0]);
	printf("0predict = %lf", output[Tensor(0)]);
	output = net.Predict(*input_learning_layers[1]);
	printf("1predict = %lf\n", output[Tensor(0)]);
	for (int n = 0; n < 1000; n++)
	{
		net.Learn(input_learning_layers, target_learning_layers, optimizer, NeuralNetwork::ErrorType::CrossEntropy);
		output = net.Predict(*input_learning_layers[0]);
		printf("0predict = %lf, %lf", output[Tensor(0)], output[Tensor(1)]);
		output = net.Predict(*input_learning_layers[1]);
		printf("1predict = %lf, %lf\n", output[Tensor(0)], output[Tensor(1)]);
	}
	for (int n = 0; n < input_learning_layers.size(); n++)
	{
		output = net.Predict(*input_learning_layers[n]);
		double target_value = 0;
		double predict = 0;
		double max_percentage = 0;
		vector<double> predict_percent;
		for (const pair<Tensor, double>& tensor : output)
		{
			double current_value = target_learning_layers[n]->GetNodeValue(tensor.first);
			if (current_value == 1)
			{
				target_value = tensor.first.GetXYChannel()[0];
			}
			double percent = tensor.second;
			if (tensor.second > max_percentage)
			{
				predict = tensor.first.GetXYChannel()[0];
				max_percentage = tensor.second;
			}

			predict_percent.push_back(tensor.second);
		}
		printf("target = %lf, predict = %lf, %lf \n", target_value, predict_percent[0], predict_percent[1]);
	}
}


int main()
{
	/*
	GateTest();
	LearnTest();
	printf("State End\n");
	QLearningTestStateEnd();
	printf("Reward End\n");
	QLearningTestRewardEnd();
	printf("QLearning End\n");
	OneHotEncodingTest();
	CrossEntropyTest();
	Layer2DTest();
	*/
	//ConvTest();
	//CrossTest();
	MiniCrossTest();
	//MnistTest();
}