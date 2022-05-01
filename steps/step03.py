#!/usr/bin/env python3
import numpy as np


class Variable:
    """DeZeroの変数クラス."""

    def __init__(self, data):
        """Initialize."""
        self.data = data


class Function:
    """基底クラス.すべての関数に共通する機能を実装する."""

    def __call__(self, input):
        """入出力はVariableインスタンスとする."""
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        """実装は継承先で行う."""
        raise NotImplementedError()


class Square(Function):
    """Square function."""

    def forward(self, x):
        """Specific calculation."""
        return x ** 2


class Exp(Function):
    """Exponential function."""

    def forward(self, x):
        """Specific calculation."""
        return np.exp(x)


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)
