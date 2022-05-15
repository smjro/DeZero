#!/usr/bin/env python3
"""関数の使いやすさ改善.

pythonの関数を使う感覚で使用できるようにする.
ndarray以外を使用したときにエラーを返す処理を追加.
"""

import numpy as np


class Variable:
    """DeZeroの変数クラス."""

    def __init__(self, data):
        """Initialize."""
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None  # 微分値
        self.creator = None  # どの関数から生み出されたかを記憶する

    def set_creator(self, func):
        """creatorを設定する."""
        self.creator = func

    def backward(self):
        """逆伝搬をループで実行."""
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    """基底クラス.すべての関数に共通する機能を実装する."""

    def __call__(self, input):
        """入出力はVariableインスタンスとする."""
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)   # 出力変数に生みの親を覚えさせる
        self.input = input         # 入力された変数を覚える
        self.output = output       # 出力を記憶させる
        return output

    def forward(self, x):
        """実装は継承先で行う."""
        raise NotImplementedError()

    def backward(self, gy):
        """逆伝搬."""
        raise NotImplementedError()


class Square(Function):
    """Square function."""

    def forward(self, x):
        """順伝搬の実装."""
        return x ** 2

    def backward(self, gy):
        """逆伝搬の実装."""
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    """Exponential function."""

    def forward(self, x):
        """順伝搬の実装."""
        return np.exp(x)

    def backward(self, gy):
        """逆伝搬の実装."""
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def as_array(x):
    """Numpyのndarray型に統一させる関数."""
    if np.isscalar(x):
        return np.array(x)
    return x


def square(x):
    """Square function."""
    f = Square()
    return f(x)


def exp(x):
    """Exponential function."""
    f = Exp()
    return f(x)


x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)
