#include "Tensor.h"


bool Tensor::operator<(const Tensor& rhs) const
{
	if (bias != rhs.bias)
	{
		return bias < rhs.bias;
	}
	else if (y != rhs.y)
	{
		return y < rhs.y;
	}

	return x < rhs.x;
}

bool Tensor::operator!=(const Tensor& rhs) const
{
	if (bias != rhs.bias)
	{
		return bias != rhs.bias;
	}
	else if (y != rhs.y)
	{
		return y != rhs.y;
	}

	return x != rhs.x;
}

bool Tensor::operator==(const Tensor& rhs) const
{
	return bias == rhs.bias && y == rhs.y && x == rhs.x;
}

bool Tensor::CheckBias() const
{
	return bias;
}

Tensor Tensor::GetBias(int channel)
{
	return Tensor(0, 0, true);
}

bool TensorConnection::operator<(const TensorConnection& rhs) const
{
	if (next != rhs.next)
	{
		return next < rhs.next;
	}

	return previous < rhs.previous;
}

Tensor TensorConnection::GetPrevious() const
{
	return previous;
}

Tensor TensorConnection::GetNext() const
{
	return next;
}
