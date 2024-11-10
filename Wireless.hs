module Sigmoid where


import Data.Maybe (fromMaybe)
import Data.Ratio

-- Define weights and bias
wOne1_1 = 2/3
wOne1_2 = 1/6

wOne2_1 = -1/3
wOne2_2 = 1/8

wTwo1_1 = 1/2
wTwo1_2 = -1/5

wTwo2_1 = 2/3
wTwo2_2 = 1/6

wThree1_1 = -1/4
wThree1_2 = 1/3

-- Backward LiRPA: 
-- f3: 0.4182519252028747, 0.5783764705106437

-- Forward LiRPA:
-- f3: 0.3438768659164918, 0.6692646649546782
-- f21: -0.9583333333333334, 1.075
-- f22: -1.1319444444444444, 1.3958333333333333
-- f11: -1.5, 1.8333333333333333
-- f12: -0.7916666666666666, 1.0416666666666667
-- f01: -2.0, 2.0
-- f02: -1.0, 3.0

-- wOne1_1 = 2
-- wOne1_2 = 1

-- wOne2_1 = -3
-- wOne2_2 = 4

-- wTwo1_1 = 4
-- wTwo1_2 = -2

-- wTwo2_1 = 2
-- wTwo2_2 = 1

-- wThree1_1 = -2
-- wThree1_2 = 1

-- f3: 4.186393589313198e-51, 0.9999999999999873
-- f21: 0.0, 48.0
-- f22: -20.0, 32.0
-- f11: -5.0, 7.0
-- f12: -10.0, 18.0
-- f01: -2.0, 2.0
-- f02: -1.0, 3.0

-- forward
fZero1 = (-2,2) 
fZero2 = (-1,3)

-- Forward LiRPA 
sOne1 =
  (wOne1_1 `lbmul` fZero1 + wOne1_2 `lbmul` fZero2,
   wOne1_1 `ubmul` fZero1 + wOne1_2 `ubmul` fZero2)

sOne2 =
  (wOne2_1 `lbmul` fZero1 + wOne2_2 `lbmul` fZero2,
   wOne2_1 `ubmul` fZero1 + wOne2_2 `ubmul` fZero2)

fOne1 =
  (alphaLbRelu sOne1 `lbmul` sOne1 + betaLbRelu sOne1,
   alphaUbRelu sOne1 `ubmul` sOne1 + betaUbRelu sOne1)

fOne2 = 
  (alphaLbRelu sOne2 `lbmul` sOne2 + betaLbRelu sOne2,
   alphaUbRelu sOne2 `ubmul` sOne2 + betaUbRelu sOne2)    

-- Linear relaxation for fOne1
-- alphaLbOne1 * sOne1 + betaLbOne1 <= fOne1 <= alphaUbOne1 * sOne1 + betaUbOne1
alphaLbOne1 = alphaLbRelu sOne1
betaLbOne1 = betaLbRelu sOne1

alphaUbOne1 = alphaUbRelu sOne1
betaUbOne1 = betaUbRelu sOne1

-- Linear relaxation for fOne2
-- alphaLbOne2 * sOne2 + betaLbOne2 <= fOne2 <= alphaUbOne2 * sOne2 + betaUbOne2
alphaLbOne2 = alphaLbRelu sOne2
betaLbOne2 = betaLbRelu sOne2

alphaUbOne2 = alphaUbRelu sOne2
betaUbOne2 = betaUbRelu sOne2


sTwo1 = 
  (wTwo1_1 `lbmul` fOne1 + wTwo1_2 `lbmul` fOne2,
   wTwo1_1 `ubmul` fOne1 + wTwo1_2 `ubmul` fOne2)

sTwo2 =
  (wTwo2_1 `lbmul` fOne1 + wTwo2_2 `lbmul` fOne2,
   wTwo2_1 `ubmul` fOne1 + wTwo2_2 `ubmul` fOne2)

fTwo1 =
  (alphaLbRelu sTwo1 `lbmul` sTwo1 + betaLbRelu sTwo1,
   alphaUbRelu sTwo1 `ubmul` sTwo1 + betaUbRelu sTwo1)

fTwo2 = 
  (alphaLbRelu sTwo2 `lbmul` sTwo2 + betaLbRelu sTwo2,
   alphaUbRelu sTwo2 `ubmul` sTwo2 + betaUbRelu sTwo2)   

-- Linear relaxation for fTwo1
-- alphaLbTwo1 * sTwo1 + betaLbTwo1 <= fTwo1 <= alphaUbTwo1 * sTwo1 + betaUbTwo1 
alphaLbTwo1 = alphaLbRelu sTwo1
betaLbTwo1 = betaLbRelu sTwo1 

alphaUbTwo1 = alphaUbRelu sTwo1
betaUbTwo1 = betaUbRelu sTwo1 

-- Linear relaxation for fTwo2
-- alphaLbTwo2 * sTwo2 + betaLbTwo2 <= fTwo2 <= alphaUbTwo2 * sTwo2 + betaUbTwo2
alphaLbTwo2 = alphaLbRelu sTwo2
betaLbTwo2 = betaLbRelu sTwo2

alphaUbTwo2 = alphaUbRelu sTwo2 
betaUbTwo2 = betaUbRelu sTwo2 

sThree1 = 
  (wThree1_1 `lbmul` fTwo1 + wThree1_2 `lbmul` fTwo2,
   wThree1_1 `ubmul` fTwo1 + wThree1_2 `ubmul` fTwo2)

fThree1 =
  (alphaLbSigmoid sThree1 `lbmul` sThree1 + betaLbSigmoid sThree1,
   alphaUbSigmoid sThree1 `ubmul` sThree1 + betaUbSigmoid sThree1)

-- Linear relaxation for fThree1 
-- alphaLbThree1 * sThree1 + betaLbThree1 <= fThree1 <= alphaUbThree1 * sThree1 + betaUbThree1
alphaLbThree1 = alphaLbSigmoid sThree1
betaLbThree1 = betaLbSigmoid sThree1

alphaUbThree1 = alphaUbSigmoid sThree1
betaUbThree1 = betaUbSigmoid sThree1

-- Backward LiRPA (m <=> w, p <=> b)

-- Given alphaLbThree1 * sThree1 + betaLbThree1 <= fThree1 
--         <= alphaUbThree1 * sThree1 + betaUbThree1
--       and
--       sThree1 = wThree1_1 * fTwo1 + wThree1_2 * fTwo2
-- replace the occurrence of sThree1 with the above expression. 
-- We get
--    mLbTwo1 * fTwo1 + mLbTwo2 * fTwo2 + pLbTwo
--     <= fThree1 <= mUbTwo1 * fTwo1 + mUbTwo2 * fTwo2 + pUbTwo ----- (1)

mLbTwo1 = alphaLbThree1 * wThree1_1
mLbTwo2 = alphaLbThree1 * wThree1_2 
pLbTwo = alphaLbThree1 * 0 + betaLbThree1 

mUbTwo1 = alphaUbThree1 * wThree1_1
mUbTwo2 = alphaUbThree1 * wThree1_2
pUbTwo = alphaUbThree1 * 0 + betaUbThree1

-- 
-- Given alphaLbTwo1 * sTwo1 + betaLbTwo1 <= fTwo1 <= alphaUbTwo1 * sTwo1 + betaUbTwo1,
--       alphaLbTwo2 * sTwo2 + betaLbTwo2 <= fTwo2 <= alphaUbTwo2 * sTwo2 + betaUbTwo2,
--       sTwo1 = wTwo1_1 * fOne1 + wTwo1_2 * fOne2, and
--       sTwo2 = wTwo2_1 * fOne1 + wTwo2_2 * fOne2
-- replace the occurrence of sTwo1 and sTwo2 with the above expressions. Then we get
--       alphaLbTwo1 * (wTwo1_1 * fOne1 + wTwo1_2 * fOne2) + betaLbTwo1 <= fTwo1 <=
--       alphaUbTwo1 * (wTwo1_1 * fOne1 + wTwo1_2 * fOne2) + betaUbTwo1
--       alphaLbTwo2 * (wTwo2_1 * fOne1 + wTwo2_2 * fOne2) + betaLbTwo2 <= fTwo2 <=
--       alphaUbTwo2 * (wTwo2_1 * fOne1 + wTwo2_2 * fOne2) + betaUbTwo2
--
-- We can simplify the above inequalities to
--       (alphaLbTwo1 * wTwo1_1) * fOne1 + (alphaLbTwo1 * wTwo1_2) * fOne2 + (alphaLbTwo1 * 0) + betaLbTwo1 <= fTwo1 <=
--       (alphaUbTwo1 * wTwo1_1) * fOne1 + (alphaUbTwo1 * wTwo1_2) * fOne2 + (alphaUbTwo1 * 0) + betaUbTwo1
--       (alphaLbTwo2 * wTwo2_1) * fOne1 + (alphaLbTwo2 * wTwo2_2) * fOne2 + (alphaLbTwo2 * 0) + betaLbTwo2 <= fTwo2 <=
--       (alphaUbTwo2 * wTwo2_1) * fOne1 + (alphaUbTwo2 * wTwo2_2) * fOne2 + (alphaUbTwo2 * 0) + betaUbTwo2
--         ----- (2)
--
-- By combining (1) and (2), we want to get 
--   mLbOne1 * fOne1 + mLbOne2 * fOne2 + pLbOne <= fThree1 <= mUbOne1 * fOne1 + mUbOne2 * fOne2 + pUbOne.  ----- (3)
--
-- We can get the following equations

wwTwo1_1 = (alphaLbTwo1 * wTwo1_1, alphaUbTwo1 * wTwo1_1)
wwTwo1_2 = (alphaLbTwo1 * wTwo1_2, alphaUbTwo1 * wTwo1_2)
wwTwo2_1 = (alphaLbTwo2 * wTwo2_1, alphaUbTwo2 * wTwo2_1)
wwTwo2_2 = (alphaLbTwo2 * wTwo2_2, alphaUbTwo2 * wTwo2_2)
ppTwo1 = (alphaLbTwo1 * 0 + betaLbTwo1, alphaUbTwo1 * 0 + betaUbTwo1)
ppTwo2 = (alphaLbTwo2 * 0 + betaLbTwo2, alphaUbTwo2 * 0 + betaUbTwo2)

mLbOne1 = mLbTwo1 `lbmul` wwTwo1_1 + mLbTwo2 `lbmul` wwTwo2_1
mLbOne2 = mLbTwo1 `lbmul` wwTwo1_2 + mLbTwo2 `lbmul` wwTwo2_2
pLbOne  = mLbTwo1 `lbmul` ppTwo1 + mLbTwo2 `lbmul` ppTwo2 + pLbTwo

mUbOne1 = mUbTwo1 `ubmul` wwTwo1_1 + mUbTwo2 `ubmul` wwTwo2_1
mUbOne2 = mUbTwo1 `ubmul` wwTwo1_2 + mUbTwo2 `ubmul` wwTwo2_2
pUbOne  = mUbTwo1 `ubmul` ppTwo1 + mUbTwo2 `ubmul` ppTwo2 + pUbTwo

-- 
-- Given alphaLbOne1 * sOne1 + betaLbOne1 <= fOne1 <= alphaUbOne1 * sOne1 + betaUbOne1,
--       alphaLbOne2 * sOne2 + betaLbOne2 <= fOne2 <= alphaUbOne2 * sOne2 + betaUbOne2,
--       sOne1 = wOne1_1 * fZero1 + wOne1_2 * fZero2, and
--       sOne2 = wOne2_1 * fZero1 + wOne2_2 * fZero2
-- replace the occurrence of sOne1 and sOne2 with the above expressions. Then we get
--       alphaLbOne1 * (wOne1_1 * fZero1 + wOne1_2 * fZero2) + betaLbOne1 <= fOne1 <=
--       alphaUbOne1 * (wOne1_1 * fZero1 + wOne1_2 * fZero2) + betaUbOne1
--       alphaLbOne2 * (wOne2_1 * fZero1 + wOne2_2 * fZero2) + betaLbOne2 <= fOne2 <=
--       alphaUbOne2 * (wOne2_1 * fZero1 + wOne2_2 * fZero2) + betaUbOne2
--
-- We can simplify the above inequalities to
--       (alphaLbOne1 * wOne1_1) * fZero1 + (alphaLbOne1 * wOne1_2) * fZero2 + (alphaLbOne1 * 0) + betaLbOne1 <= fOne1 <=
--       (alphaUbOne1 * wOne1_1) * fZero1 + (alphaUbOne1 * wOne1_2) * fZero2 + (alphaUbOne1 * 0) + betaUbOne1
--       (alphaLbOne2 * wOne2_1) * fZero1 + (alphaLbOne2 * wOne2_2) * fZero2 + (alphaLbOne2 * 0) + betaLbOne2 <= fOne2 <=
--       (alphaUbOne2 * wOne2_1) * fZero1 + (alphaUbOne2 * wOne2_2) * fZero2 + (alphaUbOne2 * 0) + betaUbOne2
--         ----- (4)
--
-- By combining (3) and (4), we want to get
--   mLbZero1 * fZero1 + mLbZero2 * fZero2 + pLbZero <= fThree1 <= mUbZero1 * fZero1 + mUbZero2 * fZero2 + pUbZero.  ----- (5)
--
-- We can get the following equations
wwOne1_1 = (alphaLbOne1 * wOne1_1, alphaUbOne1 * wOne1_1)
wwOne1_2 = (alphaLbOne1 * wOne1_2, alphaUbOne1 * wOne1_2)
wwOne2_1 = (alphaLbOne2 * wOne2_1, alphaUbOne2 * wOne2_1)
wwOne2_2 = (alphaLbOne2 * wOne2_2, alphaUbOne2 * wOne2_2)
ppOne1 = (alphaLbOne1 * 0 + betaLbOne1, alphaUbOne1 * 0 + betaUbOne1)
ppOne2 = (alphaLbOne2 * 0 + betaLbOne2, alphaUbOne2 * 0 + betaUbOne2)

mLbZero1 = mLbOne1 `lbmul` wwOne1_1 + mLbOne2 `lbmul` wwOne2_1
mLbZero2 = mLbOne1 `lbmul` wwOne1_2 + mLbOne2 `lbmul` wwOne2_2
pLbZero  = mLbOne1 `lbmul` ppOne1 + mLbOne2 `lbmul` ppOne2 + pLbOne

mUbZero1 = mUbOne1 `ubmul` wwOne1_1 + mUbOne2 `ubmul` wwOne2_1
mUbZero2 = mUbOne1 `ubmul` wwOne1_2 + mUbOne2 `ubmul` wwOne2_2
pUbZero  = mUbOne1 `ubmul` ppOne1 + mUbOne2 `ubmul` ppOne2 + pUbOne

lowerBoundBackward = mLbZero1 `lbmul` fZero1 + mLbZero2 `lbmul` fZero2 + pLbZero
upperBoundBackward = mUbZero1 `ubmul` fZero1 + mUbZero2 `ubmul` fZero2 + pUbZero


-- Interval bounds
lb (a,_) = a
ub (_,b) = b

lbmul coeff interval = 
  coeff * (if coeff > 0 
            then lb interval 
            else ub interval)

ubmul coeff interval = 
  coeff * (if coeff > 0 
            then ub interval 
            else lb interval)

-- Define the ReLU function
relu :: Rational -> Rational
relu x
  | x <= 0    = 0
  | otherwise = x

-- Define alpha and beta for ReLU
alphaUbRelu interval 
  | lb interval > 0 && ub interval > 0 = 1
  | lb interval < 0 && ub interval < 0 = 0
  | lb interval < 0 && ub interval > 0 = 
      ub interval / (ub interval - lb interval)

betaUbRelu interval
  | lb interval > 0 && ub interval > 0 = 0
  | lb interval < 0 && ub interval < 0 = 0
  | lb interval < 0 && ub interval > 0 = 
      - ub interval / (ub interval - lb interval) * lb interval


alphaLbRelu interval
  | lb interval > 0 && ub interval > 0 = 1
  | lb interval < 0 && ub interval < 0 = 0
  | lb interval < 0 && ub interval > 0 = 
    if abs(lb interval) < abs (ub interval) then 1 else 0

betaLbRelu interval = 0     
  

-- Define the sigmoid function and its derivative
sigmoid :: Rational -> Rational
sigmoid y = 1 / (1 + exp' (-y))
  where exp' z = toRational (exp (fromRational z))

sigmoidDerivative :: Rational -> Rational
sigmoidDerivative x = sigmoid x * (1 - sigmoid x)

-- Define the tangent line function passing 
-- through (x0, sigmoid x0) with slope m
tangentLine :: Rational -> Rational -> Rational
tangentLine x x0 = sigmoid x0 + (sigmoidDerivative x0) * (x - x0)

-- Define the binary search to find the x-coordinate 
-- where the tangent meets sigmoid
--
-- e.g.,  binarySearchForTangent (-116) 0 32 0.001
--        binarySearchForTangent 32 (-116) 0 0.001
--
binarySearchForTangent :: Rational -> Rational -> Rational -> 
  Rational -> Rational
binarySearchForTangent x0 low high epsilon
  | high - low < epsilon = 
      mid                -- stop when interval is smaller than epsilon
  | tangentLine mid x0 > sigmoid mid = 
      binarySearchForTangent x0 low mid epsilon  -- search left half
  | tangentLine mid x0 < sigmoid mid = 
      binarySearchForTangent x0 mid high epsilon -- search right half
  | otherwise =  mid      -- when tangent line and sigmoid meet 
                          -- within epsilon tolerance
  where
    mid = (low + high) / 2

delta = 0.001

-- Define alpha and beta for Sigmoid
alphaUbSigmoid interval
  | lb interval > 0 && ub interval > 0 =
      let mid = (lb interval + ub interval) / 2 in
        sigmoidDerivative mid
  | lb interval < 0 && ub interval < 0 =
      (sigmoid (ub interval) - sigmoid (lb interval)) 
        / (ub interval - lb interval) 
  | lb interval < 0 && ub interval > 0 =
      let d = binarySearchForTangent 
                (lb interval) 0 (ub interval) delta in
        sigmoidDerivative d 

betaUbSigmoid interval
  | lb interval > 0 && ub interval > 0 =
      let mid = (lb interval + ub interval) / 2 in
        - mid * alphaUbSigmoid interval + sigmoid mid
  | lb interval < 0 && ub interval < 0 =
      alphaUbSigmoid interval * (-lb interval) + sigmoid (lb interval)
  | lb interval < 0 && ub interval > 0 =
      let d = binarySearchForTangent 
                (lb interval) 0 (ub interval) delta in
        - d * alphaUbSigmoid interval + sigmoid d

alphaLbSigmoid interval
  | lb interval > 0 && ub interval > 0 =
      (sigmoid (ub interval) - sigmoid (lb interval)) 
        / (ub interval - lb interval)
  | lb interval < 0 && ub interval < 0 =
      let mid = (lb interval + ub interval) / 2 in
        sigmoidDerivative mid
  | lb interval < 0 && ub interval > 0 =
      let d = binarySearchForTangent 
                (ub interval) (lb interval) 0 delta in
        sigmoidDerivative d

betaLbSigmoid interval
  | lb interval > 0 && ub interval > 0 =
      alphaLbSigmoid interval * (-ub interval) + sigmoid (ub interval)
  | lb interval < 0 && ub interval < 0 =
      let mid = (lb interval + ub interval) / 2 in
        - mid * alphaLbSigmoid interval + sigmoid mid
  | lb interval < 0 && ub interval > 0 =
      let d = binarySearchForTangent 
                (ub interval) (lb interval) 0 delta in
        - d * alphaLbSigmoid interval + sigmoid d

main :: IO ()
main =
  do let (baklb, bakub) = (lowerBoundBackward, upperBoundBackward)
     putStrLn "Backward LiRPA: " 
     putStrLn ("f3: " ++ show (fromRational baklb :: Double) ++ ", "
                      ++ show (fromRational bakub :: Double))
    
     putStrLn ""

     putStrLn "Forward LiRPA: "
     
     let (fwdlb31, fwdub31) = fThree1
     putStrLn ("f3: " ++ show (fromRational fwdlb31 :: Double) ++ ", "
                      ++ show (fromRational fwdub31 :: Double))

     let (fwdlb21, fwdub21) = fTwo1
     putStrLn ("f21: " ++ show (fromRational fwdlb21 :: Double) ++ ", "
                       ++ show (fromRational fwdub21 :: Double))

     let (fwdlb22, fwdub22) = fTwo2
     putStrLn ("f22: " ++ show (fromRational fwdlb22 :: Double) ++ ", "
                       ++ show (fromRational fwdub22 :: Double))

     let (fwdlb11, fwdub11) = fOne1
     putStrLn ("f11: " ++ show (fromRational fwdlb11 :: Double) ++ ", "
                       ++ show (fromRational fwdub11 :: Double))

     let (fwdlb12, fwdub12) = fOne2
     putStrLn ("f12: " ++ show (fromRational fwdlb12 :: Double) ++ ", "
                       ++ show (fromRational fwdub12 :: Double))

     let (fwdlb01, fwdub01) = fZero1
     putStrLn ("f01: " ++ show (fromRational fwdlb01 :: Double) ++ ", "
                       ++ show (fromRational fwdub01 :: Double))

     let (fwdlb02, fwdub02) = fZero2
     putStrLn ("f02: " ++ show (fromRational fwdlb02 :: Double) ++ ", "
                       ++ show (fromRational fwdub02 :: Double))

-- -- Main function to find x where tangent touches the sigmoid function at upper bound point
-- findTangentTouchingPointUpperBound :: Rational -> Maybe Rational
-- findTangentTouchingPointUpperBound x0 = binarySearchForTangent x0 (sigmoid x0) 0 32 0.001

-- findTangentTouchingPointLowerBound :: Rational -> Maybe Rational
-- findTangentTouchingPointLowerBound x0 = binarySearchForTangent x0 (sigmoid x0) (-116) 0 0.001

