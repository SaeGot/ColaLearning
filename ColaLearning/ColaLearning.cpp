#include <stdio.h>
#include "ColaLearning.h"
int main()
{
	double input;
	printf_s("가중치를 입력하세요 : ");
	scanf_s("%lf", &input);

	Weight W(input);
	double weight = W.GetWeight();
	printf_s("%lf", weight);
}