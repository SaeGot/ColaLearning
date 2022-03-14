#pragma once


class Tensor
{
public:
	Tensor() { x = 0; y = 0; bias = false; }
	Tensor(int _x, int _y = 0, bool _bias = false) { x = _x; y = _y; bias = _bias; }

	bool operator<(const Tensor& rhs) const;
	bool operator!=(const Tensor& rhs) const;
	bool CheckBias() const;
	/**
	 * 편향 텐서 가져오기.
	 * 
	 * \param channel : 채널
	 * \return 편향 텐서
	 */
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
	/**
	 * 이전 층 텐서 가져오기.
	 * 
	 * \return 이전 층 텐서
	 */
	Tensor GetPrevious() const;
	/**
	 * 다음 층 텐서 가져오기.
	 * 
	 * \return 다음 층 텐서
	 */
	Tensor GetNext() const;

private:
	Tensor previous;
	Tensor next;
};
