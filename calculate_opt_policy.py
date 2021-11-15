import argparse
from fractions import Fraction
from decimal import Decimal
import numpy as np
from scipy.optimize import linprog
#import cvxpy as cp
import math

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=20, type=int)
    parser.add_argument("--d0", default=5, type=int)
    parser.add_argument("--M", default=20, type=float)
    return parser.parse_args()

def opt_tabular(probs):
    n = probs.shape[0]
    Q = [[[Fraction(0), Fraction(0)], [Fraction(0), Fraction(0)]] for _ in range(n + 1)]
    V = [[Fraction(0), Fraction(0)] for _ in range(n + 1)]

    Q[n][0][0] = Fraction(-1)
    Q[n][0][1] = Fraction(-1)
    V[n][0] = Fraction(-1)

    Q[n][1][0] = Fraction(1)
    Q[n][1][1] = Fraction(-1)
    V[n][1] = Fraction(1)

    prob_max = Fraction(1)

    ret = [n]

    for i in range(n - 1, 0, -1):
        p_i = Fraction(Decimal.from_float(np.float(probs[i])))
        prob_max *= 1 - p_i

        Q[i][0][0] = Fraction(-1)
        Q[i][0][1] = (1 - p_i) * V[i + 1][0] + Fraction(p_i) * V[i + 1][1]
        
        Q[i][1][0] = 2 * prob_max - 1
        Q[i][1][1] = Q[i][0][1]

        for j in range(2):
            V[i][j] = max(Q[i][j][0], Q[i][j][1])

        if V[i][1] == Q[i][1][0]:
            ret.append(i)
    
    return ret

'''
def opt_loglinear(n, d0, M, th=math.exp(-1)):
    d = d0 * 2
    A_ub = np.zeros((2 * n, d))
    b_ub = np.full((2 * n,), -M)

    for i in range(n):
        f = (i + 1) / n
        for j in range(d0):
            A_ub[i, 2 * j] = f ** j
            A_ub[n + i, 2 * j] = f ** j
            A_ub[n + i, 2 * j + 1] = f ** j
        if f > th:
            A_ub[n + i, :] *= -1

    x = cp.Variable(d)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, np.eye(d))), [A_ub @ x <= b_ub])
    prob.solve(solver=cp.OSQP, max_iter=1000000, verbose=False)

    return prob.status == "optimal", x.value
'''

if __name__ == "__main__":
    args = get_args()
    #opt_tabular(args.n)
    print(opt_loglinear(args.n, args.d0, args.M))
