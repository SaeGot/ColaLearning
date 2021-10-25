#include "ColaLearning.h"


int main()
{
	vector<double> input = {2, 5};

	Weight W(input);
	double weight1 = W.GetWeight(0);
	double weight2 = W.GetWeight(1);
	printf_s("%lf, %lf", weight1, weight2);
}