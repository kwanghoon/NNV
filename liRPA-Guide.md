# PyTorch and LiRPA Installation and Example Guide

## Prerequisites

### Step 1: Install PyTorch

To install PyTorch, use the following command:

```shell
pip install torch==2.2.2
```

### Step 2: Install LiRPA

Clone the LiRPA repository and install it as follows:

```shell
git clone https://github.com/Verified-Intelligence/auto_LiRPA
cd auto_LiRPA
pip install -e .
```

#### Note:
If you encounter encoding errors on Windows, modify the `setup.py` file as follows:

1. Replace:
   ```python
   with open('auto_LiRPA/__init__.py') as file:
   ```
   With:
   ```python
   with open('auto_LiRPA/__init__.py', encoding="utf-8") as file:
   ```

2. Replace:
   ```python
   long_description = (this_directory / 'README.md').read_text()
   ```
   With:
   ```python
   long_description = (this_directory / 'README.md').read_text(encoding='utf8')
   ```

## Running Simple Examples

### Example 1: Running Toy Example

Execute the following command:

```shell
D:\auto_LiRPA> python .\examples\simple\toy.py
```

#### Expected Output:

```
Model prediction: -1.0
IBP bounds: lower=-6.0, upper=4.0
CROWN bounds: lower=-3.0, upper=3.0
CROWN linear (symbolic) bounds: lA x + lbias <= f(x) <= uA x + ubias, where
{'lA': tensor([[[-0.3333, -0.3333]]]), 'uA': tensor([[[-1.3333,  0.3333]]]), 'lbias': tensor([[-2.]]), 'ubias': tensor([[1.3333]]), 'unstable_idx': None}
alpha-CROWN bounds: lower=-3.0, upper=2.0
alpha-CROWN linear (symbolic) bounds: lA x + lbias <= f(x) <= uA x + ubias, where
{'lA': tensor([[[-0.3333, -0.3333]]]), 'uA': tensor([[[-0.1077, -0.2795]]]), 'lbias': tensor([[-2.]]), 'ubias': tensor([[1.3333]])}
```

### Example 2: Running Vision Verification Example

Execute the following command:

```shell
D:\auto_LiRPA> python .\examples\vision\simple_verification.py
```

#### Expected Output:

```
...
```
