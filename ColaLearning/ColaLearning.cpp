#include <stdio.h>
#include "ColaLearning.h"
int main()
{
	double input;
	printf_s("����ġ�� �Է��ϼ��� : ");
	scanf_s("%lf", &input);

	Weight W(input);
	double weight = W.GetWeight();
	printf_s("%lf", weight);
}