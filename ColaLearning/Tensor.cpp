#include "Tensor.h"


bool Tensor::operator<(const Tensor& rhs) const
{
	if (x != rhs.x)
	{
		return x < rhs.x;
	}

	return y < rhs.y;
}