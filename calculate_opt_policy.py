import argparse
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
import cvxpy as cp
import math

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=20, type=int)
    parser.add_argument("--d0", default=5, type=int)
    parser.add_argument("--M", default=20, type=float)
    return parser.parse_args()

def opt_tabular(n):
    Q = [[[Fraction(0), Fraction(0)], [Fraction(0), Fraction(0)]] for _ in range(n + 1)]
    V = [[Fraction(0), Fraction(0)] for _ in range(n + 1)]

    Q[n][0][0] = Fraction(-1)
    Q[n][0][1] = Fraction(-1)
    V[n][0] = Fraction(-1)

    Q[n][1][0] = Fraction(1)
    Q[n][1][1] = Fraction(-1)
    V[n][1] = Fraction(1)

    for i in range(n - 1, 0, -1):
        Q[i][0][0] = Fraction(-1)
        Q[i][0][1] = Fraction(i, i + 1) * V[i + 1][0] + Fraction(1, i + 1) * V[i + 1][1]
        
        Q[i][1][0] = Fraction(2 * i - n, n)
        Q[i][1][1] = Q[i][0][1]

        for j in range(2):
            V[i][j] = max(Q[i][j][0], Q[i][j][1])

        print(float(V[i][1]), V[i][1])

def opt_loglinear(n, d0, M):
    d = d0 * 2
    #c = np.zeros((d,))
    A_ub = np.zeros((2 * n, d))
    b_ub = np.full((2 * n,), -M)

    for i in range(n):
        f = (i + 1) / n
        for j in range(d0):
            A_ub[i, 2 * j] = f ** j
            A_ub[n + i, 2 * j] = f ** j
            A_ub[n + i, 2 * j + 1] = f ** j
        if f > math.exp(-1):
            A_ub[n + i, :] *= -1
    
    #print("A_ub", A_ub, "b_ub", b_ub)

    #res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=(-1000, 1000))

    #return res.success, res.x

    x = cp.Variable(d)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, np.eye(d))),
                    [A_ub @ x <= b_ub])
    prob.solve(solver=cp.OSQP, max_iter=1000000, verbose=False)

    return prob.status == "optimal", x.value

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution corresponding to the inequality constraints is")
    print(prob.constraints[0].dual_value)

if __name__ == "__main__":
    args = get_args()
    #opt_tabular(args.n)
    print(opt_loglinear(args.n, args.d0, args.M))
