import unittest
import numpy as np 
from rnn_basics import *
from rnn_lstm import *


class TestRNN(unittest.TestCase):
    def test_cell_forward(self):
        print("========[Testing cell_forward]======")
        np.random.seed(1)
        xt_tmp = np.random.randn(3,10)
        a_prev_tmp = np.random.randn(5,10)
        parameters_tmp = {}
        parameters_tmp['Waa'] = np.random.randn(5,5)
        parameters_tmp['Wax'] = np.random.randn(5,3)
        parameters_tmp['Wya'] = np.random.randn(2,5)
        parameters_tmp['ba'] = np.random.randn(5,1)
        parameters_tmp['by'] = np.random.randn(2,1)

        a_next_tmp, yt_pred_tmp, cache_tmp = rnn_cell_forward(xt_tmp, a_prev_tmp, parameters_tmp)
        print("a_next[4] = \n", a_next_tmp[4])
        print("a_next.shape = \n", a_next_tmp.shape)
        print("yt_pred[1] =\n", yt_pred_tmp[1])
        print("yt_pred.shape = \n", yt_pred_tmp.shape)
    
    def test_forward(self):
        print("========[Testing forward]======")
        np.random.seed(1)
        x_tmp = np.random.randn(3,10,4)
        a0_tmp = np.random.randn(5,10)
        parameters_tmp = {}
        parameters_tmp['Waa'] = np.random.randn(5,5)
        parameters_tmp['Wax'] = np.random.randn(5,3)
        parameters_tmp['Wya'] = np.random.randn(2,5)
        parameters_tmp['ba'] = np.random.randn(5,1)
        parameters_tmp['by'] = np.random.randn(2,1)

        a_tmp, y_pred_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, parameters_tmp)
        print("a[4][1] = \n", a_tmp[4][1])
        print("a.shape = \n", a_tmp.shape)
        print("y_pred[1][3] =\n", y_pred_tmp[1][3])
        print("y_pred.shape = \n", y_pred_tmp.shape)
        print("caches[1][1][3] =\n", caches_tmp[1][1][3])
        print("len(caches) = \n", len(caches_tmp))
    
    def test_lstm_cell_forward(self):
        print("========[Testing lstm_cell_forward]======")
        np.random.seed(1)
        xt_tmp = np.random.randn(3,10)
        a_prev_tmp = np.random.randn(5,10)
        c_prev_tmp = np.random.randn(5,10)
        parameters_tmp = {}
        parameters_tmp['Wf'] = np.random.randn(5, 5+3)
        parameters_tmp['bf'] = np.random.randn(5,1)
        parameters_tmp['Wi'] = np.random.randn(5, 5+3)
        parameters_tmp['bi'] = np.random.randn(5,1)
        parameters_tmp['Wo'] = np.random.randn(5, 5+3)
        parameters_tmp['bo'] = np.random.randn(5,1)
        parameters_tmp['Wc'] = np.random.randn(5, 5+3)
        parameters_tmp['bc'] = np.random.randn(5,1)
        parameters_tmp['Wy'] = np.random.randn(2,5)
        parameters_tmp['by'] = np.random.randn(2,1)

        a_next_tmp, c_next_tmp, yt_tmp, cache_tmp = lstm_cell_forward(xt_tmp, a_prev_tmp, c_prev_tmp, parameters_tmp)
        print("a_next[4] = \n", a_next_tmp[4])
        print("a_next.shape = ", c_next_tmp.shape)
        print("c_next[2] = \n", c_next_tmp[2])
        print("c_next.shape = ", c_next_tmp.shape)
        print("yt[1] =", yt_tmp[1])
        print("yt.shape = ", yt_tmp.shape)
        print("cache[1][3] =\n", cache_tmp[1][3])
        print("len(cache) = ", len(cache_tmp))
    
    def test_lstm_forward(self):
        print("========[Testing lstm_forward]======")
        np.random.seed(1)
        x_tmp = np.random.randn(3,10,7)
        a0_tmp = np.random.randn(5,10)
        parameters_tmp = {}
        parameters_tmp['Wf'] = np.random.randn(5, 5+3)
        parameters_tmp['bf'] = np.random.randn(5,1)
        parameters_tmp['Wi'] = np.random.randn(5, 5+3)
        parameters_tmp['bi']= np.random.randn(5,1)
        parameters_tmp['Wo'] = np.random.randn(5, 5+3)
        parameters_tmp['bo'] = np.random.randn(5,1)
        parameters_tmp['Wc'] = np.random.randn(5, 5+3)
        parameters_tmp['bc'] = np.random.randn(5,1)
        parameters_tmp['Wy'] = np.random.randn(2,5)
        parameters_tmp['by'] = np.random.randn(2,1)

        a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, parameters_tmp)
        print("a[4][3][6] = ", a_tmp[4][3][6])
        print("a.shape = ", a_tmp.shape)
        print("y[1][4][3] =", y_tmp[1][4][3])
        print("y.shape = ", y_tmp.shape)
        print("caches[1][1][1] =\n", caches_tmp[1][1][1])
        print("c[1][2][1]", c_tmp[1][2][1])
        print("len(caches) = ", len(caches_tmp))


if __name__ == '__main__':
    unittest.main()