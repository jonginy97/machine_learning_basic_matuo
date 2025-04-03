# 📘 Lecture 03: Summary of PyTorch Basics

## 1. Overview of PyTorch
- A **dynamic computation graph-based** deep learning framework developed by Facebook
- Constructs the computation graph at runtime, unlike TensorFlow and Theano
- Strong in natural language processing and handling unstructured data

## 2. Basics of Tensors

### 2.1 Tensor Creation
- Created using `torch.tensor`, `torch.ones`, `torch.zeros`, `torch.full`, `torch.eye`, etc.
- Can also be created from NumPy arrays

### 2.2 Specifying Data Types
- `dtype=torch.float`, `torch.int`, `torch.long`, etc.

### 2.3 Checking Size
- `.size()`, `.shape`  
- To get a specific dimension: `.size(dim)`

### 2.4 Tensor Transformation
- `.reshape()`, `.view()`, `.unsqueeze()`, `.squeeze()`
- `.transpose()`, `.permute()`, `.split()`, `.chunk()`
- `.cat()`, `.stack()`

### 2.5 Operations
- Arithmetic operations: `+`, `-`, `*`, `/`, `log`, `exp`, `sqrt`
- Aggregate operations: `sum`, `mean`, `var`, `std`, `max`
- Norm: `torch.norm`
- Matrix multiplication: `dot`, `matmul`, `bmm`, `einsum`
- Conditions: `torch.where`, `torch.clamp`, `torch.eq`, `torch.gt`, etc.

### 2.6 Conversions
- `.numpy()`, `.tolist()`, `.item()`

### 2.7 Device Setting
- `.to(device)`, `.cuda()`, `.cpu()`
- `torch.device("cuda" if torch.cuda.is_available() else "cpu")`

## 3. Autograd (Automatic Differentiation)
- Automatic differentiation is computed when `.requires_grad = True` is set
- Perform backpropagation with `.backward()`
- Check gradients with `.grad`
- Calculations can be blocked with `.detach()`, `torch.no_grad()`

## 4. Constructing Neural Network Models (`nn.Module`)
- Layer composition by inheriting `nn.Module`
- Define by overriding `__init__` and `forward()`
- Continuous networks can be defined with `nn.Sequential`

## 5. Optimization (`torch.optim`)
- Various optimizers provided such as `optim.SGD`, `Adam`, etc.
- Learning proceeds in the order: `.zero_grad()` → `.backward()` → `.step()`

## 6. Model Training Flow
1. Data loading (e.g., `torchvision.datasets`)
2. Model definition (`MLP`, `Sequential`, etc.)
3. Loss function definition (e.g., cross-entropy)
4. Optimizer setup
5. Training loop
   - Forward pass
   - Loss calculation
   - Backward pass
   - Parameter update
6. Validation loop (for model evaluation)

## 7. Saving and Loading Models
- Saving: `torch.save(model.state_dict(), "model.pth")`
- Loading:
  ```python
  model.load_state_dict(torch.load("model.pth"))
  ```

## 8. `torchvision` & `DataLoader`
- Datasets: `torchvision.datasets.MNIST`, `CIFAR10`, etc.
- Preprocessing: `transforms.Compose([...])`
- Mini-batch processing: `torch.utils.data.DataLoader(...)`