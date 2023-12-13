module XorNN where

w1_1_1 = 3.4243
w1_2_1 = 3.4299
b1_1   = -5.3119

w1_1_2 = 4.4863
w1_2_2 = 4.4830
b1_2   = -1.7982

w2_1   = -7.1722
w2_2   = 6.7997
b2     = -3.0611


sigmoid x = 1 / (1 + exp (-x))

g1_1 x1 x2 = x1 * w1_1_1 + x2 * w1_2_1 + b1_1
g1_2 x1 x2 = x1 * w1_1_2 + x2 * w1_2_2 + b1_2
g    f1 f2 = f1 * w2_1   + f2 * w2_2   + b2

f1_1 x1 x2 = sigmoid (g1_1 x1 x2)
f1_2 x1 x2 = sigmoid (g1_2 x1 x2)

f_xor x1 x2 =
  sigmoid (f1_1 x1 x2 * w2_1 + f1_2 x1 x2 * w2_2 + b2 )
    
