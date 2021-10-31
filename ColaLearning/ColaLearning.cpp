#include "ColaLearning.h"


int main()
{
	vector<double> input = { 1, 2, 5 };
	vector<double> weight = { 0.35, 0.25, 0.08 };
	double output_sum = 0.0;
	double output_activated = 0.0;

	Node N(input);
	Weight W(weight);

	// 신호 합치기
	for (int n = 0; n < input.size(); n++)
	{
		output_sum += N.GetNodeValue(n) * W.GetWeight(n);
	}
	printf_s("신호 합친 후 값 = %lf\n", output_sum);

	// 신호 활성
	if (output_sum >= 1)
	{
		output_activated = 1.0;
	}
	else
	{
		output_activated = 0.0;
	}
	printf_s("신호 활성 후 값 = %lf\n", output_activated);
}