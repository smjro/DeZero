#!/usr/bin/env python3
"""メモリ使用量の削減.

- 不要な微分を保持しない
- 推論時は途中計算結果を保持しない
"""

import numpy as np
import unittest
import weakref
import contextlib


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


@contextlib.contextmanager
def using_config(name, value):
    """with文を使ったモード切り替え."""
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    """using_configの呼び出しを簡単にする関数."""
    return using_config('enable_backprop', False)


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
        self.generation = 0

    def set_creator(self, func):
        """creatorを設定する."""
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        """逆伝搬をループで実行."""
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):  # inputsとgxsを対応付けて処理する
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

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

        # 逆伝搬をするかどうかでモードを切り替える
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)   # 出力変数に生みの親を覚えさせる
                self.inputs = inputs         # 入力された変数を覚える
                self.outputs = [weakref.ref(output) for output in outputs]

        # リストの要素が１つのときは最初の要素を返す
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        """実装は継承先で行う."""
        raise NotImplementedError()

    def backward(self, gy):
        """逆伝搬."""
        raise NotImplementedError()


class Config:
    """Configuration."""

    enable_backprop = True


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


x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()

print(y.grad, t.grad)
print(x0.grad, x1.grad)
