#!/usr/bin/env python3
"""テスト機能の追加.

python -m unittest **.py
と入力することでテストモードとして実行できる.
"""

import numpy as np
import unittest


class SquareTest(unittest.TestCase):
    """Square関数のテスト."""

    def test_forward(self):
        """順伝搬のテスト."""
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        """逆伝搬のテスト."""
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        """勾配確認による自動テスト."""
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)  # x.gradとnum_gradが近い値かどうかを判定
        self.assertTrue(flg)


class Variable:
    """DeZeroの変数クラス."""

    def __init__(self, data):
        """Initialize."""
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None     # 微分値
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


def numerical_diff(f, x, eps=1e-4):
    """中心差分近似."""
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return ((y1.data - y0.data) / (2 * eps))


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
