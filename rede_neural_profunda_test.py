import numpy as np
import unittest
from rede_neural_profunda import *
import sklearn.datasets
from typing import Dict



            
class TestFuncaoAtivacao(unittest.TestCase):

    def test_sigmoid(self):
        a = sigmoid.dz_ultima_camada(np.array([-0.8,-0.4,0.2]),np.array([-0.8,-0.4,0.2]),np.array([0,10,1]),np.array([0,10,1]))
        arr_esp = [-0.8,-10.4,-0.8]
        for i,esp in enumerate(arr_esp):
            self.assertAlmostEqual(arr_esp[i],a[i],msg="Funcao dz_ultima_camada não está com o resultado correto para a função de ativação sigmoid!")

        lst_dz = sigmoid.dz_funcao(np.array([-0.4,0.1,1]),None,None,np.array([2,3,-1]))
        arr_esp = np.array([-1.12,0.27,0])
        for i,esp in enumerate(arr_esp):
            self.assertAlmostEqual(arr_esp[i],lst_dz[i],msg="Funcao dz_funcao não está com o resultado correto para a função de ativação sigmoid!")

    def test_relu(self):
        self.assertListEqual(list(relu.funcao(np.array([10,-4,0.2,0]))),list(np.array([10,0,0.2,0])))
        lst_dz = relu.dz_funcao(np.array([]),np.array([10,-4,0.2,0]),np.array([]),np.array([2,2,2,2]))
        self.assertListEqual(list(lst_dz),list(np.array([2,0,2,2])))

    def test_leaky_relu(self):
        self.assertListEqual(list(leaky_relu.funcao(np.array([10,-4,0.2,0]))),list(np.array([10,-0.04,0.2,0])))

        lst_dz = leaky_relu.dz_funcao(np.array([]),np.array([10,-4,0.2,0]),np.array([]),np.array([2,2,2,2]))
        self.assertListEqual(list(lst_dz),list(np.array([2,0.02,2,2])))

    def test_tanh(self):
        arr_a = tanh.funcao(np.array([10,-4,0.2,0])) #,np.array([10,-0.04,0.2,0])
        expected = [0.9999999958776927,-0.999329299739067,0.197375320224904,0]
        for i,a in enumerate(arr_a):
            self.assertAlmostEqual(a,expected[i],msg="Valor inesperado para o calculo da tangente hiperbolica (tanh)")

        arr_dz = tanh.dz_funcao(np.array([]),np.array([10,-4,0.2,0]),np.array([]),np.array([2,2,2,2]))
        expected = [1.64892291e-08, 2.68190137e-03, 1.92208597e+00, 2.00000000e+00]
        for i,dz in enumerate(arr_dz):
            self.assertAlmostEqual(dz,expected[i],msg="Valor inesperado para o calculo da funcao dz da hiperbolica (tanh)")
            
            
class TestCamada(unittest.TestCase):
    arr_entrada = [np.array([#1a entrada
                        [1,     0.75],
                        [1,     -1],
                        [0.5,   0.75],
                        [1,     1],
                    ]),
                    np.array([#Saida 1o relu
                        [1.575,	1.125,	1.175],
                        [0.7,	0,	1],
                        [1.475,	1.075,	0.725],
                        [1.7,	1.3,	1.2],
                    ]),
                    np.array([#saida 1o sigmoid
                            [0.8284952300245991,	0.7549149868676283,	0.7640475997974661],
                            [0.668187772168166,	0.47502081252106,	0.7310585786300049],
                            [0.8138161720984914,	0.7455466141264027,	0.6737070994545215],
                            [0.845534734916465,	0.785834983042559,	0.768524783499018],
                    ])
                ]
    arr_saida_sigmoid = [np.array([#entrada->sigmoid
                            [0.8284952300245991,	0.7549149868676283,	0.7640475997974661],
                            [0.668187772168166,	0.47502081252106,	0.7310585786300049],
                            [0.8138161720984914,	0.7455466141264027,	0.6737070994545215],
                            [0.845534734916465,	0.785834983042559,	0.768524783499018],

                        ]),
                        np.array([#relu->sigmoid
                            [0.962312109491394,	0.973852340311992],
                            [0.933391964424909,	0.908877038985144],
                            [0.951662371280095,	0.962039158004388],
                            [0.965443769713724, 0.97833173368304],
                        ]),
                        np.array([#sigmoid->sigmoid
                            [0.941265465941803, 0.925075768232473],
                            [0.933495907232613,	0.904592654379243],
                            [0.938386491888863,	0.919952690447872],
                            [0.942085040810341,	0.9270612845566]
                        ])]
    arr_saida_relu = [
                        np.array([#entrada->relu
                                    [1.575,	1.125,	1.175],
                                    [0.7,	0, 1],
                                    [1.475,	1.075,	0.725],
                                    [1.7,	1.3,	1.2],
                        ]),
                        np.array([#relu->relu
                                    [3.24,	3.6175],
                                    [2.64,	2.3],
                                    [2.98,	3.2325],
                                    [3.33,	3.81],

                        ]),
                        np.array([#sigmoid->relu
                            [2.77419734196394,	2.51339828596337],
                            [2.64167308750495,	2.24932916310249],
                            [2.72328076838488,	2.44170441600912],
                            [2.78911983364557,	2.54240009992864],

                        ])

                    ]
    arr_pesos_camada = [np.array([#camada 1, três unidades
                        [0.2,0.5],
                        [0.1,0.7],
                        [0.9,0.1]
                        ]),
                        np.array([#camada 2, duas unidades
                        [0.2,0.3,0.5],
                        [1,0.3,0.6]
                        ]),
                        np.array([#camada 2, duas unidades
                        [0.2,0.3,0.5],
                        [1,0.3,0.6]
                        ])
                        ]
    arr_b_camada = np.array([[1,0.5,0.2],#camada um (um b por unidade)
                    [2,1,0.5],#camada 2 (um b por unidade)
                    [2,1,0.5],#camada 2 (um b por unidade)
                    ])
    arr_unid_camada = np.array([3,2])


    def test_init(self):
        arrUnidades         = [1,3,2]
        arrUnidadesAnterior = [2,5,1]
        for i,qtd_unidades in enumerate(arrUnidades):
            c = Camada(qtd_unidades,lambda z:1,lambda a,z,y,w:1)
            self.assertEqual(len(c.arr_unidades),qtd_unidades,"A quantidade de unidades inesperado!")
            for unid in c.arr_unidades:
                self.assertTrue(np.sum(unid.arr_w)!=0,"Os pesos devem iniciar de forma aleatória, caso contrário, todas as unidades irão ter o mesmo resultado")

    def test_forward_propagation(self):
        for i,mat_entrada in enumerate(TestCamada.arr_entrada):
            camada_sigmoid = Camada(TestCamada.arr_saida_sigmoid[i].shape[1],sigmoid.funcao,sigmoid.dz_funcao)
            camada_relu = Camada(TestCamada.arr_saida_sigmoid[i].shape[1],relu.funcao,relu.dz_funcao)
            for j,unidade_pesos in enumerate(TestCamada.arr_pesos_camada[i]):
                camada_sigmoid.arr_unidades[j].arr_w = unidade_pesos
                camada_sigmoid.arr_unidades[j].b = TestCamada.arr_b_camada[i][j]


                camada_relu.arr_unidades[j].arr_w = unidade_pesos
                camada_relu.arr_unidades[j].b = TestCamada.arr_b_camada[i][j]

            camada_sigmoid.forward_propagation(mat_entrada)

            camada_relu.forward_propagation(mat_entrada)

            #verifica o resultado da mat_a
            self.assertListEqual(list(camada_sigmoid.mat_a.shape), list(TestCamada.arr_saida_sigmoid[i].shape),"Dimensões incorretas da matriz de ativações (sigmoid). NUmero de unidades: "+str(len(camada_sigmoid.arr_unidades))+" Numero de instancias: "+str(mat_entrada.shape[0]))
            for j in range(camada_sigmoid.mat_a.shape[0]):
                [self.assertAlmostEqual(camada_sigmoid.mat_a[j][k],sig_esperado,msg="Resultado inesperado da matriz de ativações (sigmoid) epserado: "+str(sig_esperado)+" Matriz A de entrada: "+str(mat_entrada)) for k,sig_esperado in enumerate(TestCamada.arr_saida_sigmoid[i][j])]

            self.assertListEqual(list(camada_relu.mat_a.shape), list(TestCamada.arr_saida_relu[i].shape),"Dimensões incorretas da matriz de ativações (relu). NUmero de unidades: "+str(len(camada_relu.arr_unidades))+" Numero de instancias: "+str(mat_entrada.shape[0]))
            for j in range(camada_relu.mat_a.shape[0]):
                [self.assertAlmostEqual(camada_relu.mat_a[j][k],relu_esperado,msg="Resultado inesperado da matriz de ativações (relu) esperado: "+str(relu_esperado)+" Matriz A de entrada: "+str(mat_entrada)) for k,relu_esperado in enumerate(TestCamada.arr_saida_relu[i][j])]
    def test_mat_w(self):
        camada_sigmoid = self.inicializa_camada_dz_dw()

        self.assertListEqual(list(camada_sigmoid.mat_w.shape),[2,3],"O tamanho das dimensões de mat_w está incorreto")
        mat_w_esperado = [[1,3,5],[3,7,8]]
        for i,arr_w_i in enumerate(camada_sigmoid.mat_w):
            self.assertListEqual(list(arr_w_i),mat_w_esperado[i],"Matriz mat_w possui elementos inesperados para a unidade "+str(i))
    def test_mat_dz(self):
        camada_sigmoid = self.inicializa_camada_dz_dw()

        self.assertListEqual(list(camada_sigmoid.mat_dz.shape),[3,2],"O tamanho das dimensões de mat_dz está incorreto")
        mat_dz_esperado = [[3,1],
                            [2,3],
                            [1,10]]
        for i,arr_dz_i in enumerate(camada_sigmoid.mat_dz):
            self.assertListEqual(list(arr_dz_i),mat_dz_esperado[i],"Matriz mat_dz possui elementos inesperados para a unidade "+str(i))
    def test_mat_dz_w(self):
        camada_sigmoid = self.inicializa_camada_dz_dw()

        dot_esperado = [[6,16,23],
                        [11,27,34],
                        [31,73,85]
                        ]
        for i,arr_dz_w_i in enumerate(camada_sigmoid.mat_dz_w):
            self.assertListEqual(list(arr_dz_w_i),dot_esperado[i],"O calculo da propriedade mat_dz_w possui elementos inesperados para a unidade "+str(i))
    def inicializa_camada_dz_dw(self):
        camada_sigmoid = Camada(2,sigmoid.funcao,sigmoid.dz_ultima_camada)
        camada_sigmoid.qtd_un_camada_ant = 3
        camada_sigmoid.arr_unidades[0].arr_w = np.array([1,3,5])
        camada_sigmoid.arr_unidades[1].arr_w = np.array([3,7,8])
        camada_sigmoid.arr_unidades[0].gradiente = Gradiente(arr_dz=[3,2,1],arr_dw=[3,2],db=0)
        camada_sigmoid.arr_unidades[1].gradiente = Gradiente(arr_dz=[1,3,10],arr_dw=[30,10],db=0)
        camada_sigmoid.qtd_un_camada_ant = 3
        camada_sigmoid.mat_a = np.array([[2,1],
                                [3,2],
                                [4,3]])
        return camada_sigmoid

    def camada_teste_backward(self, camada:Camada, 
                                    mat_z:np.array,
                                    mat_w:np.array,
                                    arr_y:np.array,
                                    funcao_ativacao:FuncaoAtivacao,
                                    nome_funcao_ativacao:str,
                                    mat_a_ant:np.array,
                                    expected_grads:Dict ):

        #coloca os pesos corretos nas unidades e a matriz de ativações
        camada.mat_a = funcao_ativacao.funcao(mat_z)
        camada.qtd_un_camada_ant = mat_a_ant.shape[1]
        for i in range(2):
            camada.arr_unidades[i].arr_z = mat_z[:,i]
            camada.arr_unidades[i].arr_a = camada.mat_a[:,i]
            camada.arr_unidades[i].mat_a_ant = mat_a_ant
            camada.arr_unidades[i].arr_w = mat_w[i]

        #executa o backward propagation
        camada.backward_propagation(arr_y)

        for i in range(2):

            grads = ["db","arr_dw"]
            for grad_name in grads:
                grad_expected = expected_grads[grad_name][i]
                

                #para cada gradiente, verifica as dimensões e os resultados
                grad =  camada.arr_unidades[i].gradiente.__dict__[grad_name]


                self.verifica_grad(grad,grad_expected,grad_name,nome_funcao_ativacao)
                
    def test_backward_propagation(self):
        np.random.seed(1)
        mat_z = np.array([[3,-3],
                 [3,0],
                [-5,3],
                [1.01,0]])
        mat_a_ant = np.array([[3.00,	0.00,	3.00],
                                [3.00,	0.00,	1.00],
                                [0.00,	3.00,	2.00],
                                [1.01,	0.00,	3.00]])
        expected_sigmoid = {
                            "arr_dw":np.array([[0.6114487779164611,	0.005019638193213643,	0.005685664213718689],
                                                [-0.4656805951168249,	0.7144305951168248,	-0.48814353170560837]]),
                            "db":np.array([0.16121531345200216,-2.77555756156289E-017])
                            }
        expected_relu ={
                        "db":np.array([-0.08716334570861797,0.18400765036856515]),
                        "arr_dw":np.array([[-0.11055054,  0,   -0.48174745],
                                        [0.23882682, 0.50060338, 0.16252645],
                                        ])
                             }
        mat_w = np.array([[0.2,0.1,0.9],
                 [0.5,0.7,0.1]])

        arr_y = np.array([1,0,0,1])

        #teste camada sigmoid (sem proxima camada)
        camada_sigmoid = Camada(2,sigmoid.funcao,sigmoid.dz_ultima_camada)
        self.camada_teste_backward(camada_sigmoid,mat_z,mat_w,arr_y,sigmoid,
                                    'sigmoid',mat_a_ant,expected_sigmoid)
        #teste camada relu (com proxima camada)
        camada_relu = Camada(2,relu.funcao,relu.dz_funcao)
        camada_relu.prox_camada = camada_sigmoid
        
        self.camada_teste_backward(camada_relu,mat_z,mat_w,arr_y,relu,
                                    'relu',mat_a_ant,expected_relu)


    def verifica_grad(self,grad,grad_expected,grad_name,nom_ativacao):
        if(type(grad)==list or type(grad)==np.ndarray):
            self.assertListEqual(list(grad.shape),list(grad_expected.shape),"A dimensão do gradiente "+grad_name+" está incorreta ("+nom_ativacao+")")

            #verifica o resultado do gradiente
            for i,grad_val in enumerate(grad):
                self.assertAlmostEqual(grad_val,grad_expected[i],
                        msg=f"Valor inesperado para o gradiente {grad_name} posição {i}"+\
                             f"  ({nom_ativacao}) deveria ser: {grad_expected[i]}\n valor obtido {grad_name}: {grad} \n esperado:{grad_expected}")
        else:
            self.assertAlmostEqual(grad,grad_expected,msg="Valor inesperado para o gradiente "+grad_name+"   ("+nom_ativacao+") deveria ser: "+str(grad_expected))

            
class TestRedeNeural(unittest.TestCase):
    def assertListAlmostEqual(self,arr_a, arr_b, msg):
        for i,a in enumerate(arr_a):
            self.assertAlmostEqual(a, arr_b[i], msg=f"Erro na posição {i} da lista: {msg}")
    def test_config_rede(self):
        mat_x = TestCamada.arr_entrada[0]
        self.redeNeural  = RedeNeural([3,2],
                                [relu,sigmoid],
                                100)
        self.redeNeural.config_rede(mat_x, np.array([1,0,0,1]))
        #primeira camada sigmoid
        #segunda camada relu
        self.assertEqual(len(self.redeNeural.arr_camadas),2)
        
        #o tipo de cada dz_funcao e funcao deve ser uma função
        for i,camada in enumerate(self.redeNeural.arr_camadas):
            for j,unidade in enumerate(camada.arr_unidades):
                self.assertTrue(callable(unidade.func_ativacao),f"O atributo 'func_ativacao' da unidade {j}  camada {i} deve ser uma função python")
                self.assertTrue(callable(unidade.dz_func),f"O atributo 'dz_func' da unidade {j}  camada {i} deve ser uma função python")
           
        #para cada camada até a penultima, é necessário armazenar em camada.prox_camada a camada seguinte
        for i,camada in enumerate(self.redeNeural.arr_camadas):
            if(i<len(self.redeNeural.arr_camadas)-1 and camada.prox_camada is None):
                self.assertTrue(callable(self.redeNeural.arr_camadas),f"A proxima camada deve ser armazenada no atributo 'prox_camada' da camada {i} ")


    def test_forward_propagation(self):
        self.redeNeural.config_rede(TestCamada.arr_entrada[0],np.array([1,0,0,1]))
        self.atualiza_pesos()
        self.redeNeural.forward_propagation()
        expected_final_a = np.array([ [0.9623121094913941, 0.9738523403119914],
                            [0.9333919644249093, 0.9088770389851438],
                            [0.9516623712800949, 0.9620391580043877],
                            [0.9654437697137236, 0.9783317336830395],
                            ])
        self.assertListEqual(list(self.redeNeural.arr_camadas[1].mat_a.shape),list(expected_final_a.shape),"As dimensões da matriz de ativação (ultima camada) estão incorretas (nao foi testado as anteriores)")
        for i,arr_inst_i in enumerate(self.redeNeural.arr_camadas[1].mat_a):
            self.assertListAlmostEqual(list(arr_inst_i),list(expected_final_a[i]),"O valor para instancia "+str(i)+" da matriz de ativação da ultima camada não está correta (nao foi testado as anteriores)")

    def test_backward_propagation(self):
        self.redeNeural.config_rede(TestCamada.arr_entrada[0],np.array([1,0,0,1]))
        self.atualiza_pesos()
        self.redeNeural.forward_propagation()
        self.redeNeural.backward_propagation()

        expected_grads_por_unidade = {"arr_dw":np.array([[0.4023691244590959, -0.07128003709783343],
                                                [0.06275930383817924, 0.0998381240784498],
                                                [0.36843448245611465, -0.06960241900129413],
                                                ]),
                                    "db":np.array([0.5464155784916468,0.13452311118634733,0.5000663175114496]),
                                    }
        for i,unidade in enumerate(self.redeNeural.arr_camadas[0].arr_unidades):
            self.assertAlmostEqual(expected_grads_por_unidade["db"][i],unidade.gradiente.db,msg="O gradiente db (primeira camada, unidade "+str(i)+") está incorreto (as demais camadas não foram testadas) Unidade:\n"+str(unidade))
            self.assertListEqual(list(expected_grads_por_unidade["arr_dw"][i].shape),list(unidade.gradiente.arr_dw.shape),"As dimensões do gradiente dw (primeira camada) estão incorretas  (as demais camadas não foram testadas) Unidade:\n"+str(unidade))
            
            self.assertListAlmostEqual(list(expected_grads_por_unidade["arr_dw"][i]),list(unidade.gradiente.arr_dw),"O gradiente dw (primeira camada, unidade "+str(i)+") está incorreto (as demais camadas não foram testadas) Unidade:\n"+str(unidade))

    def test_fit(self):
        np.random.seed(0)
        #mat_x, arr_y = sklearn.datasets.make_moons(400, noise=0.05)

        mat_x =         np.array([[0.2,0.1,0.3,0.5],#instancia 1
                                        [0.1,0.5,0.1,0.1],#instancia 2
                                        [0.9,0.1,0.4,0.7],#instancia 3
                                        [0.8,0.1,0.4,0.7]#instancia 3
                                        ])
        arr_y = np.array([0,0,1,1])
        self.redeNeural  = RedeNeural([40,1],#[4,3,3,1],
                                [sigmoid,sigmoid,relu,sigmoid],
                                1000)
        self.redeNeural.fit(mat_x,arr_y,1.1)

        expected = [0.0036241470553106283,8.316717230816547e-06,
                    0.9990737642484306,0.9968156936168789]
        self.assertListAlmostEqual(expected,list(self.redeNeural.arr_camadas[1].arr_unidades[0].arr_a),"A saida final nao está com o valor previsto!")
    def test_predict(self):
        self.test_fit()
        mat_x =         np.array([[0.1,0.01,0.4,0.5],#instancia 1
                                        [0.1,0.2,0.1,0.2],#instancia 2
                                        [0.7,0.1,0.3,0.7],#instancia 3
                                        [0.6,0.1,0.5,0.4],#instancia 4,
					                    [0.1,0.01,0.4,0.5],#instancia 5
                                        ])
        arr_predicts = self.redeNeural.predict(mat_x)
        self.assertEquals(len(arr_predicts),5, "Quantidade de resultados inesperada")
        self.assertListEqual(list(self.redeNeural.predict(mat_x)),[0, 0, 1, 1, 0])
    def atualiza_pesos(self):
        #atualiza pesos de cada camada
        arr_camada_peso_idx = [0,2]
        for i,camada_idx in enumerate(arr_camada_peso_idx):
            for j,unidade_pesos in enumerate(TestCamada.arr_pesos_camada[i]):
                self.redeNeural.arr_camadas[i].arr_unidades[j].arr_w = unidade_pesos
                self.redeNeural.arr_camadas[i].arr_unidades[j].b = TestCamada.arr_b_camada[i][j]

    def setUp(self):
        self.redeNeural  = RedeNeural([3,2],
                                            [relu,sigmoid],
                                            100)
if __name__ == "__main__":
    unittest.main()
