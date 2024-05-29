from matplotlib import pyplot as plt
import numpy as np

rng = np.random.default_rng()

class Function:
    def __init__(self, func, deriv):
        self.func = func
        self.deriv = deriv

    def __call__(self, x):
        return self.func(x)

    def deriv(self, x, x0=None):
        if not x0:
            return self.deriv(x)
        self.deriv(x, x0)


class Network:
    def __init__(self, shape, activ_fn: Function):
        assert len(shape) >= 2
        self.depth = len(shape)
        self.shape = shape
        self.activ_fn = activ_fn
        self.weights = [rng.standard_normal((y, x)) for x, y in zip(shape[:-1], shape[1:])]
        self.biases = [rng.standard_normal(x) for x in shape[1:]]
        self.grad_b = [np.zeros_like(b) for b in self.biases]
        self.grad_w = [np.zeros_like(w) for w in self.weights]
        
    def forward(self, v):
        self.z_l = [None]
        self.a_l = [v]
        for w, b in zip(self.weights, self.biases):
            self.z_l.append((w @ self.a_l[-1]) + b)
            self.a_l.append(self.activ_fn(self.z_l[-1]))
        return self.a_l[-1]

    def backward(self, y: np.array, loss, grad_loss, grad_activ):
        delta = [grad_loss(y, self.a_l[-1]) * grad_activ(self.z_l[-1])]
        for z, w in zip(self.z_l[-2:0:-1], self.weights[::-1]):
            delta.append((w.transpose() @ delta[-1]) * grad_activ(z))
        delta.reverse()
        grad_b = delta
        grad_w = [a * d[:, None] for a, d in zip(self.a_l[:-1], delta)]
        for db, _db in zip(self.grad_b, grad_b):
            db += _db
        for dw, _dw in zip(self.grad_w, grad_w):
            dw += _dw

    def step(self, m, eps):
        for b, db in zip(self.biases, self.grad_b):
            b -= eps * db / m

        for w, dw in zip(self.weights, self.grad_w):
            w -= eps * dw / m

    def zero_grad(self):
        self.grad_b = [np.zeros_like(b) for b in self.biases]
        self.grad_w = [np.zeros_like(w) for w in self.weights]


def main():
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    mse = lambda x, y: np.sum((x - y) ** 2) / (2 * len(x))
    grad_mse = lambda y0, y: y - y0
    grad_sigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))
    learning_rate = 10

    f = lambda x: np.sin(x * 3)

    net = Network((1, 3, 5, 3, 1), sigmoid)

    training_data = []
    for _ in range(1000):
        x = np.array([rng.random()])
        training_data.append((x, f(x)))

    batch_size = 100
    batches = [training_data[i:i+batch_size] for i in range(0, len(training_data), batch_size)]

    losses = []
    for e in range(200):
        err = 0
        for i, mb in enumerate(batches):
            net.zero_grad()
            for x, y in mb:
                pred = net.forward(x)
                err += mse(y, pred)
                net.backward(y, mse, grad_mse, grad_sigmoid)
            net.step(batch_size, learning_rate)
        losses.append(err)
        print(f'epoch {e}: {err}')

    plt.plot(range(len(losses)), losses)
    plt.show()


    xs = np.linspace(0, 1)
    ys = f(xs)
    plt.plot(xs, ys)
    #plt.plot(xs, list(map(net.forward, map(np.array, xs))))
    preds = []
    for x in xs:
        preds.append(net.forward(np.array([x])))
    plt.plot(xs, preds)
    plt.show()
                                           

if __name__ == '__main__':
    main()
    
