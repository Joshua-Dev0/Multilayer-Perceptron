import cupy as cp

def onehot(label: int) -> cp.ndarray:
  label = int(label)

  array = cp.zeros((10, 1))
  array[label, 0] = 1.0

  return array

def softmax(Z: cp.ndarray):
  exp_Z = cp.exp(Z - cp.max(Z))
  return exp_Z / cp.sum(exp_Z)

def relu(Z: cp.ndarray):
  return cp.maximum(0, Z)

def relu_deriv(Z: cp.ndarray):
  return (Z > 0).astype(float)

def accuracy(result: cp.ndarray, onehot_label: cp.ndarray):
  prediction = cp.argmax(result)
  actual = cp.argmax(onehot_label)

  return bool(prediction == actual)

def forward_propagation(
  X: cp.ndarray,
  W1: cp.ndarray,
  B1: cp.ndarray,
  W2: cp.ndarray,
  B2: cp.ndarray,
  W3: cp.ndarray,
  B3: cp.ndarray
):
  Z1 = W1 @ X + B1
  A1 = relu(Z1)

  Z2 = W2 @ A1 + B2
  A2 = relu(Z2)

  Z3 = W3 @ A2 + B3
  A3 = softmax(Z3)

  return A3, Z3, A2, Z2, A1, Z1
