from matplotlib import pyplot
from scipy import spatial


def GetQuarterFloat(sign, exponent, mantissa):
    if exponent == 0:
        f = mantissa * (2**-3)
    else:
        f = (2**3 + mantissa) * (2**(exponent-4))

    return f if sign == 0 else -f

if __name__ == "__main__":
    positive_quarter_floats = []
    for e in range(0, 15):
        positive_quarter_floats += [GetQuarterFloat(0, e, m) for m in range(0, 7)]

    # Aufgabe Teil b
    biggest = max(positive_quarter_floats)
    smallest = min([x for x in positive_quarter_floats if x != 0])
    length = len(positive_quarter_floats)
    print("Groesste Zahl: {}, Kleinste positive Zahl: {}, Anzahl moeglicher Zahlen: {}".format(biggest, smallest, length))

    pyplot.scatter(positive_quarter_floats, [0]*length)

    # Aufgabe Teil c
    tree = spatial.KDTree(list(zip(positive_quarter_floats, [0]*length)))
    err = []
    x = []

    for n in range(0, 10000):
        xn = 10 ** ((n - 5000) * 4.6 / 5000)
        yn = tree.query((xn, 0))[0]
        x.append(xn)
        err.append(abs(yn - xn) / xn)

    pyplot.plot(x, err)
    pyplot.show()
