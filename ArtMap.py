import numpy
import pandas 
from sklearn.model_selection import train_test_split

pandas.options.display.max_columns = None
pandas.options.display.max_rows = None

dataset = pandas.read_csv('Iris.csv', sep = ',')
dataset = dataset.drop('Id', axis=1)
train, test = train_test_split(dataset, test_size = 0.5)

def matriz_caracteristica(matriz):
    matriz = numpy.asmatrix(matriz)
    return numpy.delete(matriz, 4, 1)

def matriz_categoria(matriz):
    matriz = numpy.asmatrix(matriz)
    categoria = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    b = numpy.zeros(numpy.shape(matriz[:, 4]))
    for i in range(numpy.shape(matriz[:, 4])[0]):
        b[i, 0] = categoria[matriz[i, 4]]
    return numpy.matrix(b)

def normalizar(entrada):
    somatoria = numpy.sum(entrada)
    ac = 1 - entrada
    i = numpy.concatenate((entrada, ac), 1)
    saida = numpy.abs(i / somatoria)
    return saida

def maxima_categoria(entrada):
    return numpy.where(entrada == numpy.max(entrada))[0][0]

#a = numpy.matrix("1 0; 0 1; 0.5 0.5")
#b = numpy.matrix("1; 0; 2")
a = matriz_caracteristica(train)
b = matriz_categoria(train)

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
alfa = 0.95
beta = 1
roa = 0.95
rob = 1
roab = 0.95

for x in range(na):
    # categorias
    ct = numpy.zeros((nb, 2*mb))
    for i in range(nb):
        for k in range(2*mb):
            ct[i, k] = min(ib[x, k], wb[i, k])
    ct = numpy.sum(ct, axis = 1, keepdims = True)
    tb = ct / (alfa + numpy.sum(wb[x,:]))
    
    K = maxima_categoria(tb)

    # teste de vigilância
    if (x == 0):
        tv = numpy.zeros((na, 2*mb))
    for k in range(2*mb):
        tv[x, k] = min(ib[x, k], wb[x, k])
    if (x == 0):
        tvig = numpy.zeros(na)
    tvig[x] = numpy.sum(tv[x,:]) / numpy.sum(ib[x,:])

    while (tvig[x] < rob):
        tb[K] = 0
        K = maxima_categoria(tb)
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
    J = maxima_categoria(ta)
    if (x == 0):
        tv = numpy.zeros((na, 2*ma))
    for j in range(2*ma):
        tv[x, j] = min(ia[x, j], wa[J, j])
    tvig[x] = numpy.sum(tv[x,:]) / numpy.sum(ia[x,:])

    # match tracking
    if (x == 0):
        mt = numpy.zeros((na, nb))
        mtr = numpy.zeros(na)
    for j in range(nb):
        mt[x, j] = min(yb[x, j], wab[J,j])
    mtr[x] = numpy.sum(mt[x,:]) / numpy.sum(yb[x,:])
    
    while numpy.max(mtr) < roab:
        ta[J] = 0
        J = maxima_categoria(ta)
        
        for j in range(2*ma):
            tv[x, j] = min(ia[x, j], wa[J, j])
        tvig[x] = numpy.sum(tv[x,:]) / numpy.sum(ia[x,:])
        
        while numpy.max(tvig[x]) < roa:
            ta[J] = 0;
            J = maxima_categoria(ta);
            for j in range(2*ma):
                tv[x, j] = min(ia[x,j], wa[J,j])
            tvig[x] = numpy.sum(tv[x,:]) / numpy.sum(ia[x,:])


        for j in range(nb):
            mt[x,j] = min(yb[x,j], wab[J,j])
        mtr[x] = numpy.sum(mt[x,:]) / numpy.sum(yb[x,:])
    
    yb[x,:] = 0
    yb[x, J] = 1

    # atualização dos pesos
    for j in range(2*ma):
        wa[J, j] = beta * min(ia[x,j], wa[J,j]) + (1-beta) * wa[J,j]
    
    for k in range(2*mb):
        wb[K, k] = beta * min(ib[x,k], wa[K,k]) + (1-beta) * wb[K,k]
    
    wab[J,:] = 0
    wab[J,K] = 1  

    # entrada da fase de diagnóstico
    #ad = numpy.matrix("1 0; 0 1; 0.5 0.5");
    ad = matriz_caracteristica(test)

    iad = adc = None
    if (numpy.max(numpy.abs(ad) > 1)):
        iad = normalizar(ad)
    else:
        adc = 1 - ad
        iad = numpy.concatenate((ad, adc), 1)
    
    [nd, md] = numpy.shape(ad)
    
    # matriz de atividade
    yad = numpy.zeros((nd, nd))

    # categorias
    ct = numpy.zeros((nd, 2*md))
    for i in range(nd):
        for k in range(2*md):
            ct[i, k] = min(iad[x, k], wa[i, k])
    
    ct = numpy.sum(ct, axis = 1, keepdims = True)
    tad = ct / (alfa + numpy.sum(wa[x,:]))
    
    # categoria vencedora
    D = maxima_categoria(tad)
    
    # teste de vigilância
    for k in range(2*md):
        tv[x, k] = min(iad[x, k], wa[D, k])
    tvig[x] = numpy.sum(tv[x,:]) / numpy.sum(iad[x,:])
    
    while (tvig[x] < roa):
        tad[D] = 0
        D = maxima_categoria(tad)
        for k in range(2*md):
            tv[x, k] = min(iad[x, k], wa[D, k])
        tvig[x] = numpy.sum(tv[x,:]) / numpy.sum(iad[x,:])
    
    yad[x,:] = 0
    yad[x,D] = 1  

    ybd = numpy.dot(yad, wab)

    if (x == 0):
        fim = numpy.zeros(nd)
    
    for i in range(nd):
        for j in range(na):
            if (ybd[i,j] == 1):
                fim[i] = j;
                continue

    if (x == (na-1)):
        wbd = numpy.zeros((nd, mb))
        for i in range(nd):
            for j in range(mb):
                wbd[i,j] = wb[int(fim[i]), j]

        matriz_saidas = numpy.concatenate((wbd, matriz_categoria(test), numpy.asmatrix(test[test.columns[4]]).T), axis=1)
        print(pandas.DataFrame(matriz_saidas, columns=["Saída Processada", "Saída Esperada", "Categoria Esperada"]))
        numero_acertos = 0
        for i in range(numpy.shape(wbd)[0]):
            categorias = matriz_categoria(test)
            if (wbd[i] == 1):
                numero_acertos += 1
        porcentagem_acertos = numero_acertos/numpy.shape(wbd)[0]*100
        numero_erros = numpy.shape(wbd)[0] - numero_acertos
        porcentagem_erros = 100 - porcentagem_acertos
        matriz_estatisticas = numpy.matrix([[numero_acertos, porcentagem_acertos], [numero_erros, porcentagem_erros]])
        print(pandas.DataFrame(matriz_estatisticas,columns=["Quantidade", "Porcentagem (%)"], index=["Acertos", "Erros"]))
