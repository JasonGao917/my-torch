import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.z = self.z_act(np.matmul(self.Wzx, x) + self.bzx + np.matmul(self.Wzh, self.hidden) + self.bzh)
        self.r = self.r_act(np.matmul(self.Wrx, x) + self.brx + np.matmul(self.Wrh, self.hidden) + self.brh)
        self.n = self.h_act(np.matmul(self.Wnx, x) + self.bnx + self.r * (np.matmul(self.Wnh, self.hidden) + self.bnh))
        h_t = h_prev_t * self.z + self.n * (1 - self.z)

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        s1 = (-1, 1)
        s2 = (1, -1)
        delta = np.reshape(delta, s1)
        h = np.reshape(self.hidden, s1)
        x = np.reshape(self.x, s1)
        z = np.reshape(self.z, s1)
        r = np.reshape(self.r, s1)
        n = np.reshape(self.n, s1)

        dhdz = h - n
        dLdh = delta
        dLdz = dLdh * dhdz
        dhdn = 1 - z
        dLdn = dLdh * dhdn
        dLdnn = dLdn * np.reshape(self.h_act.backward(), s1)
        self.dWnx = np.matmul(dLdnn, np.transpose(x))
        self.dWnh = dLdnn * np.matmul(r, np.transpose(h))
        self.dbnx = np.reshape(dLdnn, s2)
        self.dbnh = np.reshape(dLdnn * r, s2)
        dLdzz = dLdz * np.reshape(self.z_act.backward(), s1)
        self.dWzx = np.matmul(dLdzz, np.transpose(x))
        self.dWzh = np.matmul(dLdzz, np.transpose(h))
        self.dbzx = np.reshape(dLdzz, s2)
        self.dbzh = np.reshape(dLdzz, s2)
        dLdr = dLdnn * np.reshape(np.matmul(self.Wnh, self.hidden) + self.bnh, s1)
        dLdrr = dLdr * np.reshape(self.r_act.backward(), s1)
        self.dWrx = np.matmul(dLdrr, np.transpose(x))
        self.dWrh = np.matmul(dLdrr, np.transpose(h))
        self.dbrx = np.reshape(dLdrr, s2)
        self.dbrh = np.reshape(dLdrr, s2)
        dx = np.matmul(np.transpose(dLdrr), self.Wrx) + np.matmul(np.transpose(dLdzz), self.Wzx) + np.matmul(np.transpose(dLdnn), self.Wnx)
        dLdhp = dLdh * z
        dh_prev_t = np.matmul(np.transpose(dLdrr), self.Wrh) + np.matmul(np.transpose(dLdzz), self.Wzh) + np.matmul(np.transpose(dLdnn * r), self.Wnh) + np.transpose(dLdhp)

        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)
        return dx, dh_prev_t
