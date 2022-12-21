import numpy as np
import random
import math as mat
import copy



E = 0.000001
Its_max = 40000
alpha = 0.000000004
forecast = 4
P = 5
m = 2


S1 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
S2 = [1, 3, 9, 27, 81, 243, 729]
S3 = [2, 6, 12, 36, 72, 216, 432, 1296, 2592]


def primary_func(S, P, m, E, Its_max, alpha, forecast):
    xl = list()
    yl = list()
    x, y = make_x_y(S, P, xl, yl)
    w1, w2 = make_W(P, m)
    x = widen_M(x, m)
    w1, w2 = education(E, Its_max, x, w1, w2, y, alpha)
    out = to_forecast(w1, w2, forecast, m, x, y)
    return out

    
def make_x_y(S, p, xl, yl):
    i = 0
    lh = len(S)
    while i + p < lh:
        l = list()
        for j in range(p):
            l.append(S[j + i])
        xl.append(l)
        yl.append(S[i + p])
        i = i+1
    X = np.array(xl)
    Y = np.array(yl)
    return X, Y
    

def widen_M(x, m):
    lenghtt = len(x)
    matrix = full_O(lenghtt, m)
    x = np.append(x, matrix, axis=1)
    return x


def make_Arr(x, y):
    global alpha
    global Its_max
    X = np.zeros(x)
    for i in len(X):
        Y = np.zeros(y)
        i = i + 1 
    return X, Y

def remake_y(Y):
    Y_new = copy.deepcopy(Y)
    for i in range(len(Y[0])):
        if Y[0][i][0] < 0:
            Y_new[0][i][0] = -1
        else:
            Y_new[0][i][0] = 1
    return Y_new


def pr(Y, X):
    for j in range(len(X)):
        k = 0
        for i in range(len(Y[0])):
            if Y[0][i][0] == X[j][i][0]:
                k = k + 1
        if k == len(Y[0]):
            return True
    return False

def Y_i(y, Y_new):
    Y_i = []
    Rm = range(m)
    for i in Rm:
        y = F_act()
        y = y +1
        Y_i.append(y)
    yi= len(Y_i)
    yne= len(Y_new)
    if yi==yne:
        print("Все отлично", Y_i)
    else: 
        print("Неккоретные данные")
        print("Введите повторно, без ошибок")
        inp = input()
    return Y_i


def make_W1(w1, p):
   weight1=0
   for i in range(p + m):
        for j in range(m):
            w1[i][j] = random.random() 
    
    
def make_W2(w2):
    weight2=0
    for i in range(m):
        w2[i] = random.random()


def make_W(p, m):
    w1 = full_O(p + m, m)
    w2 = full_O(m, 1)
    make_W1(w1, p)
    make_W2(w2)
    return w1, w2


N=10

def make_M(Neur, x1, y1):
    global M
    global N
    denr = (2*m.log2(N)) 
    if Neur==N:
        M=N/denr    
    print("Value of memory=", M)


def make_alpha(M, n, p, z):            
    global N
    print("The ratio M and N is alpha")
    a= M/N
    for i in range(N):
        print("a=", a)


def F_act(x):
    lenght=len(x[0])
    for i in range(lenght):
        x[0][i] = LRELU(x[0][i])
    return x


def full_O(x, y):
    X = np.zeros((x, y))
    return X


def LRELU(x):
    return max(0.1*x, x)


def F_t(x):
    le=len(x[0])
    for i in range(le):
        x[0][i] = 1
    return x


def w1_count(w1, alpha, delta, z, w2, r1):
    r1 = z @ w1
    d = z.T @ w2.T
    v11 = delta * d 
    v1 = v11 * F_t(r1)
    w = w1 - alpha * v1
    return w


def w2_count(w2, alpha, delta, h, r2):
    r2 = h @ w2
    v22 = delta * h.T 
    v2 = v22* F_t(r2)
    w = w2 - alpha * v2
    return w


id = 1
one = 1


def education(E, n, x, w1, w2, y, alpha):
    one = 1
    current_E = 1
    k = 1
    while k <= n and E <= current_E:
        current_E = 0
        res = len(x)
        for i in range(res):
            l1 =len(x[i])
            z = full_O(one, l1)
            for j in range(l1):
                z[0][j] = x[i][j]
            r1 = z @ w1
            h = F_act(r1)
            r2 = h @ w2
            out = F_act(r2)
            delta = out - y[i]
            w1 = w1_count(w1, alpha, delta, z, w2, r1)
            w2 = w2_count(w2, alpha, delta, h, r2)
            r3 = pow(delta, 2)[0]
            r4 = r3 / 2
            current_E = current_E + r4
        print("Итерация %d: текущая ошибка%s" % (k, current_E))
        k = k + one
    return w1, w2



def p(n, Y, X):
    global N
    yes = True
    noo = False
    summ = []
    while n <=N:
        L_X = len(X)
        for j in range(L_X):
            l = 0
            Y_r = range(len(Y[0]))
            for i in Y_r:
                if Y[i][j][0] == X[0][l][0]:
                    l = l + 1
                    if l == len(Y[0]):
                        summ.append(Y[1])
                        return yes
        return noo


def S_t(X, X_new, m, n):
    X_t = []
    x=np.zeros(m, n)
    for i in range(m):
        w1_count()
        w2_count()
        X =F_t(LRELU(x)) 
        t = t +1
        X_t.append(x)
    if n<N and len(X_t)==len(X_new) :
        print(X_t)
        return True
    else: 
        print("Ошибка")
        return False


def to_forecast(w1, w2, forecast, m, x, y):
    field = np.reshape(y[-1], 1)
    X = x[-1, :-m]
    out = []
    cycl(X, w1, w2, out, field)
    return out
    

def cycl(X, w1, w2, out, field):
    amount = range(forecast)
    for i in amount:
        X = X[1:]
        tr = np.concatenate((X, field))
        X = np.concatenate((X, field))
        tr = np.append(tr, np.array([0] * m))
        h = tr @ w1
        output = h @ w2
        field = output
        out.append(output[0])


def print_S():
    global id
    global one
    for i in S:
        print(id, "---", i)
        id = id + one


def select_S(S, P, m, E, Its_max, alpha, forecast):
    print("Выберите последовательность чисел:")
    print_S()
    select_S = input()
    choice(select_S)


def choice(select_S):
    if int(select_S) == 1:
        print(primary_func(S[0], P, m, E, Its_max, alpha, forecast))
    elif int(select_S) == 2:
        print(primary_func(S[1], P, m, E, Its_max, alpha, forecast))
    elif int(select_S) == 3:
        print(primary_func(S[2], P, m, E, Its_max, alpha, forecast))
    else:
        print("Введено неверное число, выберите 1 из предложенных вариантов")
        start()


def start():
    global P
    global m
    global E
    global Its_max
    global alpha
    global forecast 
    global S1 
    global S2 
    global S3 
    print("****Predict numbers****")


if __name__ == "__main__":
    start()
    S = [S1, S2, S3]
    select_S(S, P, m, E, Its_max, alpha, forecast)