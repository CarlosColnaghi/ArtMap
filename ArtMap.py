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
yb = numpy.zeros((nb, nb))

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

    # matriz de atividade B
    yb[x,:] = 0
    yb[x, K] = 1

    # categoria Ta
    ct = numpy.zeros((na, 2*ma))
    for i in range(na):
        for j in range(2*ma):
            ct[i, j] = min(ia[x, j], wa[i, j])
    ct = numpy.sum(ct, axis = 1, keepdims = True)
    ta = ct / (alfa + numpy.sum(wa[x,:]))

    # categoria vencedora
    J = maximaCategoria(ta)
    
    tv = numpy.zeros((na, 2*ma))
    for j in range(2*ma):
        tv[x, j] = min(ia[x, j], wa[J, j])
    tvig[x] = numpy.sum(tv[x,:]) / numpy.sum(ia[x,:])

    # match tracking
    mt = numpy.zeros((na, nb))
    mtr = numpy.zeros(na)
    for j in range(nb):
        mt[x, j] = min(yb[x, j], wab[J,j])
    mtr[x] = numpy.sum(mt[x,:]) / numpy.sum(yb[x,:])
    print(mtr)