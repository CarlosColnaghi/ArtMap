import numpy

def normalizar(entrada):
    somatoria = numpy.sum(entrada)
    ac = 1 - entrada
    i = numpy.concatenate((entrada, ac), 1)
    saida = numpy.abs(i / somatoria)
    return saida

def maximaCategoria(entrada):
    return numpy.where(entrada == numpy.max(entrada))[0][0]

a = numpy.matrix("1 0; 0 1; 0.5 0.5")
b = numpy.matrix("1; 0; 1")
ia = ib = None

# normalização
if (numpy.max(numpy.abs(a) > 1)):
    ia = normalizar(a)
else:
    ac = 1 - a
    ia = numpy.concatenate((a, ac), 1)

if (numpy.max(numpy.abs(b) > 1)):
    ib = normalizar(b)
else:
    bc = 1 - b
    ib = numpy.concatenate((b, bc), 1)

# matriz de pesos
[na, ma] = numpy.shape(a)
[nb, mb] = numpy.shape(b)

wa = numpy.ones((na, 2*ma))
wb = numpy.ones((nb, 2*mb))
wab = numpy.ones((na, nb))

# matriz de ativação
ya = numpy.zeros(na)
yb = numpy.zeros(nb)

# parâmetros da rede
alfa = 0.1
beta = 0.8
roa = 0.95
rob = 1
roab = 0.95

for x in range(1):
    
    # categorias
    ct = numpy.zeros((nb, 2*mb))
    for i in range(nb):
        for k in range(2*mb):
            ct[i, k] = min(ib[x, k], wb[i, k])
    ct = numpy.sum(ct, axis = 1, keepdims = True)
    tb = ct / (alfa + numpy.sum(wb[x,:]))
    
    K = maximaCategoria(tb)

    # teste de vigilância
    tv = numpy.zeros((na, 2*mb))
    for k in range(2*mb):
        tv[x, k] = min(ib[x, k], wb[x, k])
    tvig = numpy.zeros(na)
    tvig[x] = numpy.sum(tv[x,:]) / numpy.sum(ib[x,:])

    while (tvig[x] < rob):
        tb[K] = 0
        K = maximaCategoria(tb)
        for k in range(2*mb):
            tv[x, k] = min(ib[x, k], wb[x, k])
        tvig[x] = numpy.sum(tv[x,:]) / numpy.sum(ib[x,:])
    
    print(tv)
    print(tvig)
