import random

n = 100
while True:
    p = [0]
    for i in range(n):
        p.append(0.5 * random.random())
    sum = 0
    k = 1
    for i in range(n, 0, -1):
        sum += p[i] / (1 - p[i])
        if sum >= 1:
            k = i - 1
            break
    d = 1
    mx = 0
    for i in range(1, n + 1, 1):
        d *= 2
        if i - 1 >= k:
            d *= 1 - p[i - 1]
        if d >= mx:
            mx = d
            pos = i
    
    _t = 1
    for i in range(1, k, 1):
        _t *= 1 - p[i]
    _mx = 1 / _t

    if _mx > mx:
        print(mx, k, pos, _mx, p)