import argparse
from fractions import Fraction

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=20, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    n = args.n

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
