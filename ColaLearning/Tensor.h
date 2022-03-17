#pragma once
#include <vector>
using namespace std;


class Tensor
{
public:
	Tensor() { x = 0; y = 0; channel = 0; bias = false; }
	Tensor(int _x, int _y = 0, int _channel = 0, bool _bias = false) { x = _x; y = _y; channel = _channel; bias = _bias; }

	bool operator<(const Tensor& rhs) const;
	bool operator!=(const Tensor& rhs) const;
	bool operator==(const Tensor& rhs) const;
	Tensor operator+(const Tensor& rhs) const;
	void operator+=(const Tensor& rhs);
	bool CheckBias() const;
	/**
	 * 편향 텐서 가져오기.
	 * 
	 * \param channel : 채널
	 * \return 편향 텐서
	 */
	static Tensor Bias();
	vector<int> GetXYChannel() const;
	vector<int> GetXYChannelSize() const;
	int GetBias() const;
	vector<Tensor> GetTensors() const;
	bool CheckNegative() const;
	bool CheckOver(Tensor compare) const;
	Tensor GetInvertTensor(int x_size, int y_size) const;

private:
	int x;
	int y;
	int channel;
	// 편향은 가중치용으로 사용
	bool bias;

	int GetInvert(int origin, int max) const;
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
