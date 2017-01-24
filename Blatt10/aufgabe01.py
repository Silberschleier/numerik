
def GetQuarterFloat(sign, exponent, mantissa):
    if exponent == 0:
        f = mantissa * (2**-3)
    else:
        f = (2**3 + mantissa) * (2**(exponent-4))

    return f if sign == 0 else -f

if __name__ == "__main__":
    print(GetQuarterFloat(0, 2, 3))
