#!/usr/bin/env python3
"""同じ変数を繰り返し使う."""

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
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):  # inputsとgxsを対応付けて処理する
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)

    def cleargrad(self):
        """微分を初期化."""
        self.grad = None


class Function:
    """基底クラス.すべての関数に共通する機能を実装する."""

    def __call__(self, *inputs):  # 可変朝引数
        """入出力はVariableインスタンスとする."""
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # アンパッキングでリストを展開して渡す
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)   # 出力変数に生みの親を覚えさせる
        self.inputs = inputs         # 入力された変数を覚える
        self.outputs = outputs       # 出力を記憶させる

        # リストの要素が１つのときは最初の要素を返す
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        """実装は継承先で行う."""
        raise NotImplementedError()

    def backward(self, gy):
        """逆伝搬."""
        raise NotImplementedError()


class Add(Function):
    """Addition."""

    def forward(self, x0, x1):
        """順伝搬の実装."""
        y = x0 + x1
        return y

    def backward(self, gy):
        """逆伝搬の実装."""
        return gy, gy


class Square(Function):
    """Square function."""

    def forward(self, x):
        """順伝搬の実装."""
        return x ** 2

    def backward(self, gy):
        """逆伝搬の実装."""
        x = self.inputs[0].data
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
    return Square()(x)


def exp(x):
    """Exponential function."""
    return Exp()(x)


def add(x0, x1):
    """Addition function."""
    return Add()(x0, x1)


x = Variable(np.array(2.0))
y = add(add(x, x), x)
print("y", y.data)

y.backward()
print("x.grad", x.grad)
