#!/usr/bin/env python3
"""Variableを演算子に対応させる."""

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

    __array_priority__ = 200

    def __init__(self, data, name=None):
        """Initialize."""
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None     # 微分値
        self.creator = None  # どの関数から生み出されたかを記憶する
        self.generation = 0

    def __len__(self):
        """len関数をVariableでも使用可能にする."""
        return len(self.data)

    def __repr__(self):
        """print関数に似た機能をVariableでも使用可能にする."""
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

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

    @property
    def shape(self):
        """データのshapeの取り出し."""
        return self.data.shape

    @property
    def ndim(self):
        """データの次元数の取り出し."""
        return self.data.ndim

    @property
    def size(self):
        """データの要素数の取り出し."""
        return self.data.size

    @property
    def dtype(self):
        """データの型の取り出し."""
        return self.data.dtype


class Function:
    """基底クラス.すべての関数に共通する機能を実装する."""

    def __call__(self, *inputs):  # 可変朝引数
        """入出力はVariableインスタンスとする."""
        inputs = [as_variable(x) for x in inputs]

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


class Mul(Function):
    """Multiply."""

    def forward(self, x0, x1):
        """順伝搬の実装."""
        y = x0 * x1
        return y

    def backward(self, gy):
        """逆伝搬の実装."""
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


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


class Neg(Function):
    """負数."""

    def forward(self, x):
        """順伝搬."""
        return -x

    def backward(self, gy):
        """逆伝搬."""
        return -gy


class Sub(Function):
    """引き算."""

    def forward(self, x0, x1):
        """順伝搬."""
        y = x0 - x1
        return y

    def backward(self, gy):
        """逆伝搬."""
        return gy, -gy


class Div(Function):
    """割り算."""

    def forward(self, x0, x1):
        """順伝搬."""
        y = x0 / x1
        return y

    def backward(self, gy):
        """逆伝搬."""
        x0, x1 = self.inputs[0].data, self.inputs[1]
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


class Pow(Function):
    """累乗."""

    def __init__(self, c):
        """初期化."""
        self.c = c

    def forward(self, x):
        """順伝搬."""
        y = x ** self.c
        return y

    def backward(self, gy):
        """逆伝搬."""
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
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


def as_variable(obj):
    """インスタンスをVariableに変換する関数."""
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def square(x):
    """Square function."""
    return Square()(x)


def exp(x):
    """Exponential function."""
    return Exp()(x)


def add(x0, x1):
    """Addition function."""
    x1 = as_array(x1)
    return Add()(x0, x1)


def mul(x0, x1):
    """Multiply function."""
    x1 = as_array(x1)
    return Mul()(x0, x1)


def neg(x):
    """Negative number."""
    return Neg()(x)


def sub(x0, x1):
    """Subtraction function."""
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    """Subtraction function."""
    x1 = as_array(x1)
    return Sub()(x1, x0)


def div(x0, x1):
    """Division function."""
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    """Division function."""
    x1 = as_array(x1)
    return Div()(x1, x0)


def pow(x, c):
    """Power function."""
    return Pow(c)(x)


Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__add__ = add
Variable.__radd__ = add
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow

x = Variable(np.array(2.0))

y = x ** 3

print(y)
