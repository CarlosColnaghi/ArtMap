import numpy

def normalizar(entrada):
    somatoria = numpy.sum(entrada)
    ac = 1 - entrada
    i = numpy.concatenate((entrada, ac), 1)
    saida = numpy.abs(i / somatoria)
    return saida

a = numpy.matrix("1 0; 0 1; 0.5 0.5")
b = numpy.matrix("1; 0; 1")
ia = None
if (numpy.max(numpy.abs(a) > 1)):
    ia = normalizar(a)
else:
    ac = 1 - a
    ia = numpy.concatenate((a, ac), 1)
print("Teste")