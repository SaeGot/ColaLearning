#include "ColaLearning.h"


int main()
{
	vector<double> input = { 2, 5 };
	vector<double> weight = { 1, 4 };
	vector<double> output(2);

	Node N(input);
	Weight W(weight);

	for (int n = 0; n < input.size(); n++)
	{
		output[n] = N.GetNodeValue(n) * W.GetWeight(n);
	}
	printf_s("%lf, %lf", output[0], output[1]);
}