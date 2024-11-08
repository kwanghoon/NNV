module Sigmoid where


import Data.Maybe (fromMaybe)
import Data.Ratio

-- Define weights and bias
wOne1_1 = 2
wOne1_2 = 1

wOne2_1 = -3
wOne2_2 = 4

wTwo1_1 = 4
wTwo1_2 = -2

wTwo2_1 = 2
wTwo2_2 = 1

wThree1_1 = -2
wThree1_2 = 1

-- forward
fZero1 = (-2,2) 
fZero2 = (-1,3)

--
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

alphaUbOne1 = alphaUbRelu sOne1
alphaLbOne2 = alphaLbRelu sOne2
betaUbOne1  = betaUbRelu sOne1
betaLbOne2  = betaLbRelu sOne2

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

alphaUbTwo1 = alphaUbRelu sTwo1
alphaLbTwo2 = alphaLbRelu sTwo2
betaUbTwo1  = betaUbRelu sTwo1
betaUbTwo2  = betaUbRelu sTwo2

sThree1 = 
  (wThree1_1 `lbmul` fTwo1 + wThree1_2 `lbmul` fTwo2,
   wThree1_1 `ubmul` fTwo1 + wThree1_2 `ubmul` fTwo2)

fThree1 =
  (alphaLbSigmoid sThree1 `lbmul` sThree1 + betaLbSigmoid sThree1,
   alphaUbSigmoid sThree1 `ubmul` sThree1 + betaUbSigmoid sThree1)

alphaUbThree1 = alphaUbSigmoid sThree1
alphaLbThree1 = alphaLbSigmoid sThree1
betaUbThree1  = betaUbSigmoid sThree1
betaLbThree1  = betaLbSigmoid sThree1

-- Backward LiRPA

_alphaUbThree1 = alphaUbThree1
_alphaLbThree1 = alphaLbThree1
_betaUbThree1  = betaUbThree1
_betaLbThree1  = betaLbThree1 



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
  do let x = binarySearchForTangent (-116) 0 32 0.001
     putStrLn ("Found tangent touch point at x: " ++ show x)
      
     let x = binarySearchForTangent 32 (-116) 0 0.001
     putStrLn ("Found tangent touch point at x: " ++ show x)


-- -- Main function to find x where tangent touches the sigmoid function at upper bound point
-- findTangentTouchingPointUpperBound :: Rational -> Maybe Rational
-- findTangentTouchingPointUpperBound x0 = binarySearchForTangent x0 (sigmoid x0) 0 32 0.001

-- findTangentTouchingPointLowerBound :: Rational -> Maybe Rational
-- findTangentTouchingPointLowerBound x0 = binarySearchForTangent x0 (sigmoid x0) (-116) 0 0.001

