
Ubuntu, VirtualBox, 180GB
anaconda 설치
> conda create -n marabou python=3.11.9 
 cf. maraboupy 파이썬 인터페이스는 3.11을 지원
> pip install torch
> pip install numpy


~/work/Marabou에 소스 내려 받음
현재 디렉토리: ~/work/nnv

> conda activate marabou
> pip install ../Marabou

modelpruned0.93_NO_SIGMOID.pth

> python nnv2nnet.py

wirelessModel.nnet

> python nnv.py

