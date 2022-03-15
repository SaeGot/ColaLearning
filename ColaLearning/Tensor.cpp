#include "Tensor.h"
#include <tuple>

bool Tensor::operator<(const Tensor& rhs) const
{
	return tie(y, x, bias, channel) < tie(rhs.y, rhs.x, rhs.bias, rhs.channel);
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

Tensor Tensor::operator+(const Tensor& rhs) const
{
	Tensor result;
	result.x = x + rhs.x;
	result.y = y + rhs.y;
	result.channel = channel + rhs.channel;
	if (bias)
	{
		result.bias = bias;
	}
	else if (rhs.bias)
	{
		result.bias = rhs.bias;
	}
	else
	{
		result.bias = false;
	}

	return result;
}

void Tensor::operator+=(const Tensor& rhs)
{
	Tensor result;
	x += rhs.x;
	y += rhs.y;
	channel += rhs.channel;
	if (bias)
	{
		bias;
	}
	else if (rhs.bias)
	{
		bias = rhs.bias;
	}
	else
	{
		bias = false;
	}
}

bool Tensor::CheckBias() const
{
	return bias;
}

Tensor Tensor::GetBias()
{
	return Tensor(0, 0, 0, true);
}

vector<int> Tensor::GetXYChannel() const
{
	return { x, y , channel };
}

vector<int> Tensor::GetXYChannelSize() const
{
	return { x + 1, y + 1, channel + 1 };
}

vector<Tensor> Tensor::GetTensors() const
{
	int bias_int = bias;
	vector<Tensor> tensors;
	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < y; j++)
		{
			for (int k = 0; k < channel; k++)
			{
				for (int b = 0; b < bias_int + 1; b++)
				{
					tensors.push_back(Tensor(i, j, k, b));
				}
			}
		}
	}

	return tensors;
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
