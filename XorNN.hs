module XorNN where

import Test.QuickCheck

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
  sigmoid (g (f1_1 x1 x2) (f1_2 x1 x2) )

threshold :: Double
threshold = 0.5

propRobustness :: Double -> Double -> Property
propRobustness x1 x2 =
  if 0.5 <= x1 && x1 <= 1.0 &&
     0.5 <= x2 && x2 <= 1.0 ||
     0   <= x1 && x1 <  0.5 &&
     0   <= x2 && x2 <  0.5
  then (f_xor x1 x2 < 0.5) === True
  else if x1 < 0 || 1 < x1 || x2 < 0 || 1 < x2
       then True === True
       else (f_xor x1 x2 > 0.5) === True
    
propRobustness2 :: Double -> Double -> Property
propRobustness2 x1 x2 =
  if -threshold <= x1 - x2 && x1 - x2 <= threshold
    then (f_xor x1 x2 < 0.5) === True
    else (f_xor x1 x2 > 0.5) === True

-- How to run:
--   stack ghci XorNN.hs --package QuickCheck
--
-- ghci> f_xor 0.0 0.4
-- 0.5482040334480841
-- ghci> quickCheck propRobustness
-- +++ OK, passed 100 tests.
-- ghci> quickCheck propRobustness
-- +++ OK, passed 100 tests.
-- ghci> quickCheck propRobustness
-- *** Failed! Falsified (after 3 tests and 4 shrinks):
-- 0.5
-- 0.5
-- False /= True
-- ghci> f_xor 0.5 0.5
-- 0.9136752529151544

-- ghci> quickCheck propRobustness2
-- *** Failed! Falsified (after 3 tests and 4 shrinks):
-- 1.0
-- 0.6
-- False /= True
-- ghci> f_xor 1.0 0.6
-- 0.454266900491033

-- After increasing the threshold to 0.49
--
-- ghci> quickCheck  propRobustness2
-- *** Failed! Falsified (after 2 tests and 5 shrinks):
-- 0.1
-- 0.3
-- False /= True
-- ghci> f_xor 0.1 0.3
-- 0.5483615796619974

