# Deep Learning
### TensorFlow Fundamentals | Keras Neural Networks | PyTorch & NLP Pipelines
 
**Author:** Kevin Victor
**Scope:** Consolidated reference across three laboratory modules
**Status:** Educational & Applied
 
---
 
## Module Overview
 
This module covers the complete deep learning stack — from TensorFlow's tensor primitives through the Keras high-level training API to PyTorch's explicit manual paradigm and Natural Language Processing pipelines.
 
| Module | Focus Area |
|---|---|
| **Module 10** | TensorFlow Fundamentals — Tensors, dtypes, Operations, Reshaping, Variables, GradientTape |
| **Module 11** | Keras Neural Networks — Model Compilation, `fit()`, Evaluation, Loss Curves, Applied Pipelines |
| **Module 12** | PyTorch & NLP — Manual Training Loops, Tokenization, Vectorization, Embeddings, Sentiment Analysis |
 
The unifying principle: raw data must be represented as typed tensors, fed through a well-defined computation graph, and optimized by automatic differentiation — whether expressed through TensorFlow's computational infrastructure, Keras's abstracted API, or PyTorch's explicit manual paradigm.
 
---
 
## Module 10 — TensorFlow Fundamentals
 
### Core Concepts
 
**Tensors** are the universal data structure of deep learning — multi-dimensional arrays characterized by three attributes: shape (size along each dimension), rank (number of dimensions), and dtype (element data type). A scalar is rank 0; a vector rank 1; a matrix rank 2. Every value flowing through a neural network — raw inputs, weights, activations, loss — is a tensor.
 
**Data Types (dtype)** govern what operations are valid and what precision is available. `tf.float32` is the standard for neural network weights and continuous features (sufficient precision, hardware-optimized). `tf.int32` is appropriate for class labels and indices. `tf.string` is used for unencoded text, which cannot participate in mathematical operations until encoded. dtype mismatches cause precision loss (int32 for continuous measurements) or loss function errors (float labels where integer class indices are expected).
 
**Performance** is TensorFlow's central advantage over pure Python. `tf.matmul()` calls optimized BLAS routines in C++ with hardware-level parallelism; a 300×300 matrix multiplication that takes tens of seconds in Python nested loops completes in milliseconds. This gap grows with computational complexity — it is not a convenience but a fundamental enabler of neural network training at scale.
 
**Tensor Reshaping** changes a tensor's dimensional structure without modifying its data or element count. A 1D tensor of shape `(20,)` must be reshaped to `(20, 1)` before being passed to any model, conforming to the `(samples, features)` convention. `tf.reshape()` is a view operation — it reinterprets the memory layout without copying data, making it essentially free computationally.
 
**`tf.constant` vs `tf.Variable`** defines the mutability boundary in TensorFlow. Constants are immutable — `.assign()` raises an error — appropriate for input data and fixed hyperparameters. Variables are mutable via `.assign()`, `.assign_add()`, `.assign_sub()`, and are the container for all learnable model parameters (weights, biases). Python has no equivalent enforcement; TensorFlow's distinction is architecturally enforced.
 
**In-Graph Normalization** uses `tf.reduce_mean(tensor, axis=0)` and `tf.math.reduce_std(tensor, axis=0)` to normalize feature tensors directly within a TensorFlow computation, without requiring Scikit-learn. Normalized features are extracted via slice notation (`data[:, 0]`) and combined through weighted linear expressions. `tf.where(condition, x, y)` applies decision thresholds to produce class predictions.
 
**`tf.GradientTape` and Automatic Differentiation** is the core mechanism enabling neural network training. Inside a `GradientTape` context, every operation on `tf.Variable` objects is recorded. After the forward pass and loss computation, `tape.gradient(loss, model.trainable_variables)` automatically applies the chain rule across the full computation graph, returning gradients for every trainable parameter simultaneously. `optimizer.apply_gradients()` updates each parameter using those gradients.
 
### Industrial Use Cases
 
| Domain | Pattern Applied |
|---|---|
| Battery / Materials Science | `tf.float32` continuous features + `tf.int32` labels — mirrors battery research database dtype conventions |
| Scientific Computing / HPC | `tf.matmul()` performance benchmark — the same operation is the core computation of every neural network forward pass |
| Meteorology / IoT | Tensor reshape (1D → 2D) and in-graph normalization for streaming sensor preprocessing pipelines |
| Autonomous Vehicles / ADAS | Full `GradientTape` neural network — structurally identical to perception models in production AV systems |
 
---
 
## Module 11 — Keras Neural Networks
 
### Core Concepts
 
**Model Architecture** in `tf.keras.Sequential` defines the network as a stack of layers with three decisions per layer: neuron count, layer type (Dense = fully connected), and activation function. **ReLU** (`max(0, x)`) is the standard hidden-layer activation — computationally inexpensive and resistant to vanishing gradients. **Sigmoid** maps output to (0, 1), used for binary classification output layers where the result is directly a probability. **Softmax** produces a probability distribution over K classes summing to 1, used for multi-class output layers. Without activation functions, any depth of Dense layers collapses to a single linear transformation.
 
**Model Compilation** configures the learning objective with three required settings. The **optimizer** (Adam by default) implements the gradient descent update rule with adaptive per-parameter learning rates. The **loss function** defines what is minimized: `binary_crossentropy` for sigmoid-output binary classification (requires scalar predictions in (0,1)); `sparse_categorical_crossentropy` for softmax-output multi-class classification with integer labels; `categorical_crossentropy` for softmax-output multi-class with one-hot labels. **Metrics** (`'accuracy'`) track performance without affecting the gradient. A model without compilation has architecture but no learning objective — it cannot be trained.
 
**Training with `model.fit()`** runs the gradient descent loop for a specified number of epochs, using mini-batch updates of `batch_size` samples per gradient step. With validation data provided, Keras evaluates the held-out set at each epoch end without updating weights, producing the learning trajectory in a returned `History` object with four series: `'loss'`, `'val_loss'`, `'accuracy'`, `'val_accuracy'`.
 
**Learning Curves** — plotting accuracy and loss per epoch for both training and validation — are the primary diagnostic for training behavior. Training accuracy rising while validation accuracy stagnates signals overfitting. Both curves low and flat signals underfitting. Oscillating validation metrics with smooth training curves signal too-high a learning rate. These patterns are the basis for all hyperparameter adjustment decisions.
 
**Multi-class Prediction** uses `model.predict()` to return raw output probabilities, then `np.argmax(pred_probs, axis=1)` to select the highest-probability class per sample. For binary classification with sigmoid output, the threshold is applied as `(probs > 0.5).astype(int)`.
 
**Hierarchical Feature Learning** — the reason multiple hidden layers outperform one — stems from each layer learning progressively more abstract representations: early layers detect individual feature thresholds; middle layers detect compound relationships between features; final layers abstract these into high-level patterns sufficient for classification. The progressive compression (e.g., 32 → 16 → 8 → 1) forces the network to produce increasingly distilled representations at each stage.
 
**Hybrid Rule-Model Architectures** use rule-based keyword or threshold logic for high-confidence, clear-cut inputs, and fall back to the neural network only for ambiguous cases. This pattern is more robust than either pure rules (cannot handle novel inputs) or pure models (unreliable on high-certainty cases) alone. It is the standard architecture for production intent classification and safety decision systems.
 
**Dynamic Edge Case (DEC) Labeling** captures compensatory safety logic: a risk factor may be offset by a compensating favorable factor (e.g., low wind stability compensated by high battery charge). Encoding these compensatory conditions as training labels enables the model to learn and generalize them, rather than requiring hard-coded rules for every case.
 
### Industrial Use Cases
 
| Domain | Pattern Applied |
|---|---|
| Drone Delivery / UAV Safety | Progressive pipeline: dataset → compile → train → evaluate → predict → provenance |
| Quantum Sensing / Semiconductor QC | Three hidden layers for multi-channel noise pattern detection |
| Warehouse Automation (AMR) | Four-phase robot operation: Exploration → Learning → Planning → Execution |
| Smart Home / Voice Assistants | Hybrid rule-model architecture for voice command intent classification |
| Construction Inspection / Structural Assessment | DEC labeling + sigmoid binary classification for go/no-go flight decisions |
| Humanoid Robotics | Complete Keras workflow for multi-class physical object recognition |
 
---
 
## Module 12 — PyTorch & NLP Pipelines
 
### Core Concepts
 
**PyTorch vs Keras Philosophy** is the central architectural contrast. Keras abstracts the training loop — `compile()` configures it, `fit()` executes it. PyTorch provides no equivalent abstraction. Every training step is written explicitly: forward pass, loss computation, `optimizer.zero_grad()` (clears accumulated gradients from previous steps — omitting this corrupts updates), `loss.backward()` (backpropagation through the dynamic computation graph), `optimizer.step()` (weight update). This transparency makes PyTorch the standard for research and debugging; Keras is standard for production pipelines.
 
**`nn.Module`** is the base class for all PyTorch neural networks. Subclasses define layers as instance attributes in `__init__()` and implement `forward()` to define the computation. Layers defined as attributes are automatically registered as parameter containers — their weights are included in `model.parameters()` and tracked by the autograd engine.
 
**Dataset and DataLoader** separate data access from training logic. A custom `Dataset` subclass implements `__len__()` and `__getitem__(idx)`. `DataLoader(dataset, batch_size=32, shuffle=True)` wraps it to yield `(X_batch, y_batch)` tuples. Feature engineering is placed inside the `Dataset` class — the training loop receives clean features and is responsible only for optimization.
 
**`BCEWithLogitsLoss`** is PyTorch's numerically stable binary classification loss. It combines sigmoid and binary cross-entropy in a single operation using the log-sum-exp trick, avoiding floating-point underflow when logits are large-magnitude. The model outputs raw logits (unbounded); `torch.sigmoid(output)` converts to probability only at inference time.
 
**Gradient Inspection** accesses `param.grad` after `loss.backward()` to monitor training health. Gradients near zero indicate vanishing gradients (early layers not learning); very large gradients indicate exploding gradients (numerical instability). `model.eval()` + `torch.no_grad()` disables dropout and gradient tracking during evaluation — omitting either produces incorrect evaluation results.
 
**Tokenization** is the mandatory first step in every NLP pipeline — decomposing raw text into discrete processable units. Whitespace tokenization (`sentence.split()`) is simple but ignores punctuation. Regex tokenization (`re.findall(r"\w+|[^\w\s]", text)`) separates punctuation as distinct tokens and preserves hyphenated compound terms. **Subword tokenization** decomposes rare long words into shorter, more frequent subword units, preventing out-of-vocabulary failures — the core principle behind BPE (used in GPT-4) and WordPiece (used in BERT).
 
**Vocabulary and Vectorization** assign a unique integer index to each token. `vocab.get(token, -1)` returns -1 for unknown tokens; `<UNK>` and `<PAD>` special tokens handle unknowns and fixed-length padding respectively. Padding ensures variable-length token sequences become fixed-size tensors (`max_len` truncation/padding), enabling batch stacking.
 
**Word Embeddings** replace arbitrary integer indices with dense real-valued vectors of fixed dimensionality (`nn.Embedding(vocab_size, dim)`). The embedding matrix is learned by gradient descent. Training with `CosineEmbeddingLoss` and co-occurrence pairs drives semantically related words to similar directions in vector space — after training, `torch.cosine_similarity()` retrieves nearest semantic neighbors, underpinning semantic search and analogy reasoning.
 
**Soft Labels** assign continuous values in [0, 1] proportional to semantic intensity (e.g., fraction of positive emotions in a combination) rather than hard 0/1 binary labels. Models trained on soft labels learn a graduated probability space that reflects the continuous nature of sentiment — more faithful to ambiguous or mixed emotional states than binary classification.
 
**`categorical_crossentropy` vs `sparse_categorical_crossentropy`** differ only in label format: the former requires one-hot vectors (produced by `tf.keras.utils.to_categorical(y, num_classes)`); the latter accepts integer class indices directly. Both compute the same loss. For multi-class problems, `sparse_categorical_crossentropy` eliminates the encoding step at no cost to model quality.
 
### Industrial Use Cases
 
| Domain | Pattern Applied |
|---|---|
| Climate Science / Heatwave Prediction | PyTorch `nn.Module` + Dataset/DataLoader + manual training loop for multi-scale atmospheric modeling |
| NLP / Conversational AI | Tokenization → vocabulary → vectorization → embedding pipeline — foundation of BERT, GPT, and production chatbots |
| Smart Home / Sentiment-Driven IoT | Soft-label sentiment embedding model for nuanced user intent detection |
| Geothermal Energy / Drilling Safety | TensorFlow binary classifier with domain-engineered composite features and provenance reporting |
| Semantic Search / Vector Retrieval | `CosineEmbeddingLoss` embedding training — foundation of vector database retrieval systems |
 
---
 
## Future Industry-Grade Extensions
 
The following upgrade paths apply across all three modules and represent the standard engineering investments for production deep learning systems.
 
**Data Pipeline:** Replace `tf.constant()` with `tf.data.Dataset.from_tensor_slices()` for batching, shuffling, prefetching (`tf.data.AUTOTUNE`), and on-the-fly augmentation. Use TFRecord format for large-scale training. In PyTorch, use `WeightedRandomSampler` for imbalanced datasets and pin-memory DataLoaders for GPU transfer optimization.
 
**Regularization:** Insert `tf.keras.layers.Dropout(rate=0.3)` between Dense layers to prevent overfitting — randomly zeroes a fraction of activations during training, disabled automatically during evaluation. Add `BatchNormalization()` layers to normalize activations across batches, enabling higher learning rates and accelerating convergence in deep networks. Apply `kernel_regularizer=tf.keras.regularizers.l2(0.01)` to penalize large weight magnitudes.
 
**Callbacks for Automated Training Management:** `ModelCheckpoint(save_best_only=True)` saves weights whenever validation loss improves, preserving the best-performing model rather than the final-epoch model. `EarlyStopping(patience=5)` halts training when validation loss stops improving, preventing overfitting. `ReduceLROnPlateau` halves the learning rate when training stagnates. `TensorBoard` logs loss curves, weight histograms, and computation graphs for interactive monitoring.
 
**Advanced Architectures:** Move beyond `Sequential` to the Keras **Functional API** for multi-input models, skip connections (ResNet-style residual blocks), and shared layers. Apply `nn.LSTM` for sequential data with temporal dependencies. Use `nn.MultiheadAttention` (transformer self-attention) where different parts of the input carry different relevance to the prediction — the architecture underlying BERT and GPT.
 
**Class Imbalance Handling:** Use SMOTE (`imbalanced-learn`) for oversampling minority class with synthetic interpolated examples. Apply `class_weight` in `model.fit()` or `pos_weight` in `BCEWithLogitsLoss` to upweight minority class loss contributions. Use `train_test_split(..., stratify=y)` to maintain class proportions across splits.
 
**NLP: Transfer Learning and Pretrained Embeddings:** Replace embeddings trained from scratch with pretrained GloVe (300-dimensional, 840B token corpus) or FastText (subword-aware). For sequence understanding, use HuggingFace `transformers` BERT/RoBERTa as frozen or fine-tuned sentence encoders, replacing the full tokenization-vocabulary-embedding pipeline with a single pretrained model call.
 
**NLP: Sequence-Aware Models:** Replace mean pooling over token embeddings with `nn.LSTM` (captures left-to-right dependencies) or `nn.MultiheadAttention` (captures global pairwise token relationships). LSTM enables detection of negation patterns ("not happy" ≠ "happy") that position-invariant pooling cannot represent.
 
**Explainability:** Replace threshold rule-based explanation engines with **SHAP** (`shap.DeepExplainer` for Keras, `shap.GradientExplainer` for PyTorch) — theoretically grounded per-prediction feature attribution. Use **Integrated Gradients** via `tf-explain`. Calibrate probability outputs with `CalibratedClassifierCV` for reliable confidence estimates in safety-critical decisions.
 
**Production Deployment:**
- **TensorFlow:** `model.save()` → TensorFlow Serving (REST/gRPC endpoints with model versioning); `tf.lite.TFLiteConverter` for mobile/embedded deployment (quantized int8, 4× model compression); TFX for end-to-end pipeline orchestration.
- **PyTorch:** `model.state_dict()` for parameter persistence; `torch.jit.script()` for Python-runtime-free deployment; `torch.onnx.export()` for cross-framework ONNX deployment; TorchServe for REST inference endpoints.
- **Monitoring:** Use Evidently AI or Nannyml for input distribution drift and prediction confidence degradation detection. Implement automated retraining pipelines triggered by monitoring thresholds.
**GPU and Distributed Training:** `tf.device('/GPU:0')` for explicit GPU placement; TensorFlow's `MirroredStrategy` for multi-GPU single-machine training; `MultiWorkerMirroredStrategy` for multi-machine training. Mixed-precision training (`mixed_float16` policy) doubles GPU throughput. For very large scale, use Google TPUs — purpose-built for neural network matrix multiplications at hundreds of teraflops.
 
---
 
## Concept-to-Production Mapping
 
| Demonstrated Concept | Production Equivalent |
|---|---|
| `tf.constant` / `tf.Variable` | Read-only data tensors / learnable `nn.Parameter` objects |
| `tf.matmul()` benchmark | BLAS-accelerated matrix ops in every neural network forward pass |
| `tf.reshape()` (1D → 2D) | Data pipeline input shape normalization before model inference |
| `tf.GradientTape` training loop | PyTorch autograd; TFX Trainer component |
| `model.compile()` + `model.fit()` | MLflow experiment tracking + TFX pipeline orchestration |
| Accuracy/loss learning curves | TensorBoard; W&B experiment dashboards |
| Sigmoid output + `binary_crossentropy` | Binary safety classifier in ADAS, medical screening |
| Softmax output + `sparse_categorical_crossentropy` | Multi-class intent classifier in conversational AI |
| Whitespace / regex tokenization | BPE tokenizer in GPT-4; WordPiece in BERT |
| `<UNK>` / `<PAD>` special tokens | Standard tokens in all production LLM vocabularies |
| `nn.Embedding` + `CosineEmbeddingLoss` | Word2Vec / GloVe; vector database semantic retrieval |
| Soft labels (continuous [0,1]) | Label smoothing in production classification training |
| Rule-based explanation engine | SHAP DeepExplainer; Integrated Gradients |
| `model.save()` / `state_dict()` | TensorFlow SavedModel; PyTorch TorchScript → TorchServe |
 
---
 
## Summary
 
These three modules collectively establish the full deep learning stack. Module 10 builds the foundation: tensors are typed, multi-dimensional data structures; TensorFlow executes operations on them orders of magnitude faster than Python; `tf.Variable` enables automatic differentiation; `GradientTape` makes gradient computation for millions of parameters tractable. Module 11 applies this foundation through Keras: compilation defines the learning objective, `fit()` executes the training loop, learning curves diagnose model behavior, and architectural choices — activation functions, layer count, output configuration — determine what class of functions the network can represent. Module 12 introduces a second paradigm (PyTorch's explicit manual control, preferred in research), extends deep learning to unstructured text (tokenization → vocabulary → embedding → classification), and demonstrates that the same TensorFlow binary classification workflow generalizes across physical domains from drone safety to geothermal drilling. Together, these modules establish the conceptual and practical foundation for every production AI system — from voice assistants and autonomous vehicles to scientific instruments and energy infrastructure — that relies on neural network inference.
 
---
 
 
