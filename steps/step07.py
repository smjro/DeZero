#!/usr/bin/env python3
"""バックプロパゲーションの自動化.

逆伝搬の計算を一つひとつ書いていたのを自動化する.
"""

import numpy as np


class Variable:
    """DeZeroの変数クラス."""

    def __init__(self, data):
        """Initialize."""
        self.data = data
        self.grad = None  # 微分値
        self.creator = None  # どの関数から生み出されたかを記憶する

    def set_creator(self, func):
        """creatorを設定する."""
        self.creator = func

    def backward(self):
        """逆伝搬を再帰的に実行."""
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()


class Function:
    """基底クラス.すべての関数に共通する機能を実装する."""

    def __call__(self, input):
        """入出力はVariableインスタンスとする."""
        x = input.data
        y = self.forward(x)
        output = Variable(y)
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


def f(x):
    """Composite function."""
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)
