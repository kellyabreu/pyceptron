# -*- coding: utf-8 -*-

# Um framework perceptron de múltiplas camadas escrito em Python

#Copyright (c) 2013 jpbanczek@gmail.com
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The views and conclusions contained in the software and documentation are those
#of the authors and should not be interpreted as representing official policies,
#either expressed or implied, of the FreeBSD Project.

__author__ = "Jhonathan Paulo Banczek (jpbanczek@gmail.com)"
__copyright__ = "Copyright (C) 2013 Jhonathan Paulo Banczek"
__license__ = "New BSD License"


import math
import random
import scipy


# -define as funcoes de ativacao
def func_sigmoid(value):
    """ funcao sigmoide """

    y = []
    v = 0.0

    for i in value:
        v = 1/(1+(math.exp(-i)))
        y.append(v)

    return scipy.array(y)

def func_tanh(value):
    """funcao tangente hiperbolica"""

    y = []
    v = 0.0

    for i in value:
        v = math.tanh(i)
        y.append(v)

    return scipy.array(y)

def func_gauss(value):
    """funcao gaussiana"""

    y = []
    v = 0.0
    for i in value:
        v = math.exp((-i**2))
        y.append(v)

    return scipy.array(y)

def der_sigmoid(value):
    """derivada da funcao sigmoide"""

    y = []
    v = 0.0
    for i in value:
        v = math.exp(i)/((math.exp(i+1))**2)
        y.append(v)

    return scipy.array(y)

def der_tanh(value):
    """derivada da funcao tangente"""
    y = []
    v = 0.0
    for i in value:
        v = 1 - i**2
        y.append(v)

    return scipy.array(y)

def der_gauss(value):
    """derivada da funcao gaussiana"""
    pass

#- define funcao de somatório-
def func_sum(x, w):
    """
    x,w -> vetores, faz a soma ponderada de cada elemento e
    retornar um vetor de somas para a proxima camada
    """
    output = []
    s = 0.0

    for j in range(len(w)):
        s = x * w[j]
        s = sum(s)
        output.append(s)

    return scipy.array(output)

#--preenche matriz com dados pseudo aletórios
def randvalues(arg):
    """
    arg -> tupla
    funcao que inicializa os pesos com valores aleatorios do intervalo
    (-0.2,0.2)
    """
    m = scipy.ones(arg)
    for k in range(arg[0]):
        for l in range(arg[1]):
            m[k][l] = random.uniform(-1,1)

    return m

#- - classe Mlp
class Pyceptron(object):
    """
    Framework - Multilayer Perceptron
    Framework para criação de uma rede neural de proposito geral usando 1
    ou duas camadas ocultas
    Última alteração: 17/07/2013 [ver arquivo -> CHANGES]
    atributo: self.f -> funcao de ativacao, self.dv -> derivada de self.f
    """

    def __init__(self, arg):
        """
        atributos da classe Mlp, arg -> tupla (0,0,0,0)
        ( num. neuronios da c. entrada , n. neuronios 1ª c. oculta
        n.neuronios 2ª camada oculta, n neuronios c. saida )
        """

        self.topology = arg
        self.minput = None
        self.moutput = None
        self.tx = None
        self.it = None
        self.mm = None
        self.error_m = None
        self.f = None
        self.df = None

        #camada de entrada, oculta 1, oculta 2 (opcional), camada de saida
        self.layer_input = None
        self.layer_hidden1 = None
        self.layer_hidden2 = None
        self.layer_output = None

        #constantes de momento
        self.momentum_hidden1 = None
        self.momentum_hidden2 = None
        self.momentum_output = None

    def data(self, *arg):
        """ recebe os dados de entrada e saída para treinamento.
        arg[0] -> matriz de entrada
        arg[1] -> matriz de saida esperada
        """
        self.minput = arg[0]
        self.moutput = arg[1]

    def set(self, *arg):
        """
        recebe os parametros da rede neural.
        arg[0] -> taxa de aprendizado
        arg[1] -> numero máximo de iterações,
        arg[2] -> tipo de função de ativação: str -> sigm, gauss, tanh
        arg[3] -> constante momento(momentum)
        """
        self.tx = arg[0]
        self.it = arg[1]
        #escolhe a funcao de ativacao do neuronio
        if arg[2] == 'sigm':
            self.f = func_sigmoid
        elif arg[2] == 'gauss':
            self.f = func_gauss
        elif arg[2] == 'tanh':
            self.f = func_tanh
        else:
            self.f = func_sigmoid

        #contante de momentum
        self.mm = arg[3]
        # configura a topologia da rede
        self._config_topology()

    def _config_topology(self):
        """ seta nas camadas a topologia definida no atributo self.topology
            topology -> [0] = n. neuronios camada de entrada
                        [1] = n. " na 1º camada oculta
                        [2] = n. " na 2ª camada oculta (0 -> senao tiver)
                        [3] = n. " na camada de saída
        """

        self.layer_input = scipy.zeros((self.topology[0], 1))
        self.layer_hidden1 = randvalues((self.topology[1], self.topology[0]))
        self.momentum_hidden1 = scipy.zeros(self.layer_hidden1.shape)

        #verifica se existe mais de 1 camada oculta, se existir considera ela
        if self.topology[2] >= 1:
            self.layer_hidden2 = randvalues((self.topology[2], self.topology[1]))
            #pesos da camada oculta aleatorios
            self.momentum_hidden2 = scipy.zeros(self.layer_hidden2.shape)
            self.layer_output = scipy.zeros((self.topology[3], self.topology[2]))
            self.momentum_output = scipy.zeros(self.layer_output.shape)

        else:
            self.layer_output = scipy.zeros((self.topology[3], self.topology[1]))
            self.momentum_output = scipy.zeros(self.layer_output.shape)

    def print_layers( self ):
        """
        *esse método será removido na versão final.*
        imprime os valores armazenados nas matrizes de neuronios de cada
        camada
        """

        print( "entrada -> \n", self.layer_input )
        print( 'oculta 1 -> \n', self.layer_hidden1 )
        print( 'oculta 2-> \n' , self.layer_hidden2 )
        print( 'saida -> \n', self.layer_output )
        print( 'momento hidden1 -> \n', self.momentum_hidden1 )
        print( 'momento hidden2 -> \n', self.momentum_hidden2 )
        print( 'momento output -> \n', self.momentum_output,'\n',
         "-"*40, '\n' )

    def dimension_synapses( self ):
        """ imprime a quantidade de sinapses na rede neural """

        x = None

        #verifica se existe mais de 1 camada oculta, se existir considera ela
        if self.topology[2] > 0:
            x = ((self.topology[0] * self.topology[1]) +
                (self.topology[1] * self.topology[2]) +
                (self.topology[2] * self.topology[3]))
        else:
            x = ((self.topology[0] * self.topology[1]) +
                (self.topology[1] * self.topology[3]))

        print( x, 'conexões synapticas')

    def execute( self ):
        """ inicia o treinamento da rede neural """

        #enquanto o numero de iteracoes nao atingir o máximo
        for iteration in range(self.it):

            error = 0.0
            #para cada linha da matriz de entrada e saída
            for entry, output in zip(self.minput, self.moutput):
                #print("entry: ", entry, " | saida: ", output )
                self._forward(entry)
                error += self._backward(output)
                #self.print_layers()

    def test(self):
        """ testa a rede neural """
        pass

    def _forward(self, entry):
        """ passo para frente: executa a primeira fase do treinamento da
         rede neural
         entry -> elemento da matriz de entrada
        """
        
        #verifica se existe mais de 1 camada oculta, se existir considera ela
        if self.topology[2] > 0:        
            self.layer_hidden1 = self.f(func_sum(entry, self.layer_hidden1))
            self.layer_hidden2 = self.f(func_sum(self.layer_hidden1,
                self.layer_hidden2))
            self.layer_output = self.f(func_sum(self.layer_hidden2,
                self.layer_output))

        else:
            self.layer_hidden1 = self.f(func_sum(entry, self.layer_hidden1))
            self.layer_output = self.f(func_sum(self.layer_hidden1,
                self.layer_output))



    def _backward( self, output ):
        """ passo para trás: executa a segunda fase do treinamento da rede
        neural
        output -> elemento de saida (matriz de saida esperada)
        """
        #out = saida, h2 = hidden2, h1 = hidden1, in = entrada

        delta_out_h2 = None
        delta_out_h1 = None
        delta_h2_h1 = None
        delta_h1_in = None
        #calcula os erros das camadas, começando da e saída para a de entrada
        
        #verifica se existe mais de 1 camada oculta, se existir considera ela
        if self.topology[2] > 0:
            delta_out_h2 = scipy.zeros((1, self.topology[3]))
            delta_h2_h1 = scipy.zeros((self.topology[3], self.topology[2]))
            delta_h1_in = scipy.zeros((self.topology[2], self.topology[1]))

        else:
            delta_out_h1 = scipy.zeros((1, self.topology[3]))
            # inserir as outras partes da funcao,



