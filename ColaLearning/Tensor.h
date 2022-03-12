#pragma once


class Tensor
{
public:
	Tensor(int _x, int _y = 0) { x = _x; y = _y; }

	bool operator<(const Tensor& rhs) const;

private:
	int x;
	int y;
};
