#include "Tensor.h"
#include <tuple>

bool Tensor::operator<(const Tensor& rhs) const
{
	return tie(y, x, channel, bias) < tie(rhs.y, rhs.x, rhs.channel, rhs.bias);
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

Tensor Tensor::GetInvertTensor(int x_size, int y_size) const
{
	return Tensor(GetInvert(x, x_size - 1), GetInvert(y, y_size - 1), channel, bias);
}

Tensor Tensor::Bias()
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

int Tensor::GetBias() const
{
	return bias;
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

bool Tensor::CheckNegative() const
{
	return x < 0 || y < 0 || channel < 0;
}

bool Tensor::CheckOver(Tensor compare) const
{
	return x < compare.x && y < compare.y;
}

int Tensor::GetInvert(int origin, int max) const
{
	float origin_f = float(origin);
	float max_f = float(max);

	if (origin_f - max_f / 2 < 0)
	{
		return int( origin_f + 2 * (max_f / 2 - origin_f) );
	}
	else
	{
		return int( origin_f - 2 * (origin_f - max_f / 2) );
	}
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
