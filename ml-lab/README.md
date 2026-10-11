# ML Visual Lab — Neural Network Spam Classifier

An interactive, browser-only learning tool for understanding a feedforward neural network from the inside.

The first lesson uses a tiny **spam vs not-spam** classification problem and exposes the full loop:

1. numeric inputs
2. weighted sums
3. ReLU activation
4. output logit
5. sigmoid probability
6. binary cross-entropy loss
7. backpropagation
8. gradient-descent update

## What makes this visualizer different

The network does not just draw circles and arrows:

- weights are printed directly on their connections
- during backprop, every connection shows its gradient `g = ∂L/∂w`
- during the update step, each connection also shows `Δ = -ηg`
- forward and backward signal flow is animated in opposite directions
- a parameter inspector shows value, gradient and update side-by-side
- prediction and loss are shown before and after updates

## Stack

No framework is required.

```text
index.html
styles.css
app.js
```

Everything runs client-side, making the project ideal for GitHub Pages.

## Run locally

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Deploy on GitHub Pages

1. Create a public GitHub repository.
2. Put the repository files in the root of `main`.
3. Open **Settings → Pages**.
4. Under **Build and deployment** choose:
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/(root)`
5. Save.

The page will be available at:

```text
https://YOUR_USERNAME.github.io/YOUR_REPOSITORY_NAME/
```

## Teaching network

```text
4 handcrafted email features
          ↓
3 hidden neurons + ReLU
          ↓
1 output neuron + sigmoid
          ↓
P(spam)
```

Binary cross-entropy is used as the loss, with vanilla gradient descent for parameter updates.

## Good next modules

- XOR: why hidden layers and nonlinearities matter
- activation playground: ReLU vs sigmoid vs tanh
- optimizer playground: SGD vs momentum vs Adam
- regression: MSE and gradient descent on a line
- multiclass classification: logits + softmax + cross entropy
- convolution visualizer
- attention visualizer
