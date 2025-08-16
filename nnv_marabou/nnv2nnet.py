# Ubuntu, VirtualBox, 180GB
# anaconda 설치
# conda create -n marabou python=3.11.9 
#  cf. maraboupy 파이썬 인터페이스는 3.11을 지원
# pip install torch
# pip install numpy

# For maraboupy
# pip install onnx2pytorch
# pip install onnx
# pip install onnxruntime

import torch

path = "models/modelpruned0.93_NO_SIGMOID.pth"
# path = "models/modelpruned0.93.pth"

model = torch.load(path, weights_only=False)

# numberOfAntenna = 16

# 전체 계층 구조
print("\n[Layers]")
for name, module in model.named_modules():
    print(f"{name}")

# 입력
print("\n[Input]")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"Input features: {module.in_features}")
        break  # 첫 번째 Linear 레이어의 입력이 네트워크 입력

# w와 b
print("\n[Weights and Biases]")

tensorList = []

for name, module in model.named_modules():
    if hasattr(module, 'weight') and module.weight is not None:
        print(f"{name} - weight shape: {module.weight.shape}")
        print(module.weight)
        tensorList.append(module.weight.data)

    if hasattr(module, 'bias') and module.bias is not None:
        print(f"{name} - bias shape: {module.bias.shape}")
        print(module.bias)

# print("\n[Tensor list]: ", len(tensorList))
# print(tensorList)

# 액티베이션 함수
print("\n[Activation Functions]")
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh)):
        print(f"{name} - activation: {module.__class__.__name__}")

# 배치 정규화
print("\n[Batch Normalization Layers]")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm3d):
        print(f"{name} - {module.__class__.__name__} (num_features: {module.num_features})")

# 출력
print("\n[Output]")
last_linear = None
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        last_linear = module
if last_linear is not None:
    print(f"\n[Last Layer Output]")
    print(f"Last Linear Layer Output features: {last_linear.out_features}")

# tensorList는 
# - tensor.Size(456,256) 텐서
# - tensor.Size(491,491) 텐서
# - tensor.Size(16,491) 텐서
# 3개의 텐서를 원소로 하는 리스트입니다. 

# for tensor in tensorList:
#     size1 = tensor.size[0]
#     size2 = tensor.size[1] 

# Local Marabou 빌드 설치!
# pip install ../Marabou/
# 
# GLIBCXX_3.4.32 not found 에러 발생시:
# conda install -c conda-forge libstdcxx-ng

# import sys
# sys.path.append('../Marabou')
# import maraboupy.Marabou as Marabou


# SMT 엔진을 기반으로 하는 뉴럴네트웤 검증 도구 Marabou의 제약식을 만드는 프로그램을 작성합니다. 
# 256개 입력을 받아 16개 출력을 내는 3개의 계층으로 구성된 뉴럴네트워크입니다. 
# tensorList는 3개 텐서를 원소로 하는 리스트입니다. 
# 이 3개 텐서를 아래에서 설명하는 3개 계층을 만드는데 사용합니다. 
# 첫번째 계층은 tensor.Size(491,256) 텐서로 표현한 weight를 가지고 있고, bias는 모두 0입니다.
# 뉴럴넷 256개 입력을 입력으로 하고, 491개 중간 출력을 내는 fully connected 구성입니다.
# 491개 중간 출력은 액티베이션 함수 ReLU를 통과시킨 491개 최종 출력을 냅니다. 
# 두번째 계층은 tensor.Size(491,491) 텐서로 표현한 weight를 가지고 있고, bias는 모두 0입니다. 
# 앞의 계층 491개 최종 출력을 입력으로(텐서 두번째 차원의 491에 해당) 합니다. 
# 이 입력을 받아, 491개 중간 출력을(텐서 첫번째 차원의 491에 해당) 내는 fully connected 구성입니다. 
# 491개 중간 출력은 액티베이션 함수 ReLU를 통과시킨 491개 최종 출력을 냅니다. 
# 세번째 계층은 tensor.Size(16,491) 텐서로 표현한 weight를 가지고 있고, bias는 모두 0입니다.
# 앞의 계층 491개 최종 출력을 입력으로 합니다. 
# 이 입력을 받아, 16개 최종 출력을 내는 fully connected 구성입니다. 
# 이 계층은 별도의 액티베이션 함수를 사용하지 않습니다. 

with open('wirelessModel.nnet', 'w') as f:
    # Comments
    f.write("// wirelessModel.nnet")
    f.write("\n")

    # numLayers, inputSize, outputSize, maxLayerSize
    f.write(str(3) + ", " + str(256) + ", " + str(16) + ", " + str(491) + ", ")
    f.write("\n")

    # inputSize, layerSizes, outputSize
    f.write(str(256) + ", " + str(491) + ", " + str(491) + ", " + str(16) + ", ")
    f.write("\n")

    # ReLU
    f.write(str(0) + ", ")
    f.write("\n")

    # 입력 최소, 최대, 평균, 범위
    f.write(','.join(['-1000'] * 256))
    f.write(",\n")
    f.write(','.join(['1000'] * 256))
    f.write(",\n")

     # 입력 평균, 범위 + 출력 평균, 범위
    f.write(','.join(['0.0'] * 256))
    f.write(', ' + '0.0')
    f.write(",\n")
    f.write(','.join(['1.0'] * 256))
    f.write(', ' + '1.0')
    f.write(",\n")

    # 계층1 가중치(행렬 256 x 491), 바이어스(벡터 491)
    tensor = tensorList[0]
    print(tensor.size())
    for row in tensor:
        f.write(','.join(str(x.item()) for x in row))
        f.write(",\n")

    for b in range(tensor.size(0)):
        f.write("0.0,\n")

    # 계층2 가중치(행렬 491 x 491), 바이어스(벡터 491)
    tensor = tensorList[1]
    print(tensor.size())
    for row in tensor:
        f.write(','.join(str(x.item()) for x in row))
        f.write(",\n")

    for b in range(tensor.size(0)):
        f.write("0.0,\n")

    # 계층3 가중치(행렬 491 x 16), 바이어스(벡터 16)
    tensor = tensorList[2]
    print(tensor.size())
    for row in tensor:
        f.write(','.join(str(x.item()) for x in row))
        f.write(",\n")

    for b in range(tensor.size(0)):
            f.write("0.0,\n")        




