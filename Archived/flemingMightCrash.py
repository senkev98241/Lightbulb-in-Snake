from sympy import Symbol, integrate, lambdify, pprint

def diffModel(): 
    T, a, b, o, v, r, c = (Symbol("T"), Symbol("a"), Symbol("b"), Symbol("o"), Symbol("v"), Symbol("r"), Symbol("c"))
    # diffFunc = PHO * ( (VOLT ** 2 / RESIST) - ( (a * (temp - OUTSIDETEMP) ) + (b * (temp ** 4 - OUTSIDETEMP ** 4) ) ) )
    diffFunc = 1 / ( (v ** 2 / r) - ( (a * (T - o) ) + (b * (T ** 4 - o ** 4) ) ) )
    integratedFunc = integrate(diffFunc, T) + c
    print(integratedFunc)
    return lambdify([T, a, b, o, v, r, c], expr=integratedFunc, modules="scipy", cse=True, docstring_limit=None)

diffModel()
print("Never printed anything")