# Neural Network Visualizer — Spam Classifier

A small interactive learning tool that explains a basic feedforward neural network step by step:

1. Inputs
2. Weighted sums
3. ReLU activation
4. Output logit
5. Sigmoid prediction
6. Binary cross-entropy loss
7. Backpropagation
8. Gradient descent

The first task is a tiny **spam vs not spam** classifier with four handcrafted numeric features.

## Why this repo is simple

This version intentionally uses:

- plain HTML
- plain CSS
- plain JavaScript
- no backend
- no npm dependencies
- no build step

That makes it ideal for **GitHub Pages**.

## Run locally

You can simply open `index.html`, but using a tiny local server is usually nicer:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Deploy to GitHub Pages

1. Create a new public GitHub repository.
2. Upload these files to the repository root.
3. Go to **Settings → Pages**.
4. Under **Build and deployment**, choose:
   - Source: **Deploy from a branch**
   - Branch: **main**
   - Folder: **/(root)**
5. Save.

Your site will appear at:

```text
https://YOUR_USERNAME.github.io/YOUR_REPOSITORY_NAME/
```

## Current architecture

```text
4 input features
      ↓
3 hidden neurons
      ↓
ReLU
      ↓
1 output neuron
      ↓
Sigmoid
      ↓
P(spam)
```

The model is intentionally tiny so every number can be inspected directly in the browser.

## Next ideas

- manual weight editing
- show gradients on every edge
- animate forward/backward flow
- mini training graph for loss
- XOR lesson
- softmax + multiclass lesson
- activation comparison
- optimizer comparison
- CNN convolution visualizer
- attention visualizer
