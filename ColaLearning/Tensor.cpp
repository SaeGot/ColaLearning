#include "Tensor.h"


bool Tensor::operator<(const Tensor& rhs) const
{
	if (bias != rhs.bias)
	{
		return bias < rhs.bias;
	}
	else if (channel != rhs.channel)
	{
		return channel < rhs.channel;
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
	else if (channel != rhs.channel)
	{
		return channel != rhs.channel;
	}
	else if (y != rhs.y)
	{
		return y != rhs.y;
	}

	return x != rhs.x;
}

bool Tensor::operator==(const Tensor& rhs) const
{
	return bias == rhs.bias && channel == rhs.channel && y == rhs.y && x == rhs.x;
}

bool Tensor::CheckBias() const
{
	return bias;
}

Tensor Tensor::GetBias()
{
	return Tensor(0, 0, 0, true);
}

vector<int> Tensor::GetXYChannel()
{
	return { x, y, channel };
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
