#pragma once


class Tensor
{
public:
	Tensor() { x = 0; y = 0; bias = false; }
	Tensor(int _x, int _y = 0, bool _bias = false) { x = _x; y = _y; bias = _bias; }

	bool operator<(const Tensor& rhs) const;
	bool operator!=(const Tensor& rhs) const;
	bool CheckBias() const;
	static Tensor GetBias(int channel = 0);

private:
	// 편향 존재 시 x = 최대값 +1, y = 0 이 편향 역할
	int x;
	int y;
	bool bias;
};

class TensorConnection
{
public:
	TensorConnection(Tensor _previous, Tensor _next) { previous = _previous; next = _next; }

	bool operator<(const TensorConnection& rhs) const;
	Tensor GetPrevious() const;
	Tensor GetNext() const;

private:
	Tensor previous;
	Tensor next;
};
