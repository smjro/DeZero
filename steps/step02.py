#!/usr/bin/env python3
import numpy as np


class Variable:
    """DeZeroの変数クラス."""

    def __init__(self, data):
        """Initialize."""
        self.data = data


class Function:
    """このクラス内で実装するメソッドは入出力をVariableとする."""

    def __call__(self, input):
        """すべての関数に共通する機能を実装する."""
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


x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)

#  LocalWords:  DeZeroの変数クラス
