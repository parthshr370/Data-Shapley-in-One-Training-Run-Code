
BLOG - https://medium.com/@parthshr370/data-shapley-in-one-training-run-a74c35dfcb4f 

# In-Run Data Shapley Example

This repository demonstrates **In-Run Data Shapley** computation for a single validation sample during training on the **MNIST** dataset. The code shows how to:

1. Train a model on batches of training data, while **simultaneously** computing a Shapley-like contribution score for each training sample with respect to a single validation example.
2. Use a **first-order approximation** (via per-sample gradient dot-products) to measure how much each training sample **helps** or **hurts** the validation example’s loss.

## Key Ideas

1. **Combined Loss**: We create a “combined batch” of [train_batch,val_sample] $[\text{train\_batch}, \text{val\_sample}]$ so we can compute both training and validation gradients in one forward pass.
2. **Shapley Approximation**:
    - We do a **dot product** $\langle \nabla \ell_{\text{train\_i}}, \nabla \ell_{\text{val}} \rangle$ to assess how a single training sample’s gradient aligns with (or opposes) the validation sample’s gradient.
    - We accumulate these scores in a `data_shapley_scores` tensor (one entry per training sample).
3. **Naive Implementation**: The code does multiple backward passes (one per sample in the batch) to demonstrate the concept. A true “ghost dot product” approach can do it in a single pass using forward/backward hooks, but that’s more advanced.
## How It Works

1. **Load Data**: The code loads MNIST with 60,000 training images and some validation/test set.
2. **Set Up Model & Optimizer**: A standard PyTorch model (e.g., a small CNN or MLP) plus an optimizer (e.g., `SGD`).
3. **Training Loop**:
    - For each batch, we concatenate one validation sample.
    - Compute **combined_loss** = train loss + validation loss on that single val sample.
    - **Backward** to get gradients for both train and val.
    - **Isolate** the validation gradient by a separate backward pass on just the val sample.
    - For each training sample in the batch, do a backward pass to get that sample’s gradient.
    - **Dot product** with the stored validation gradient.
    - Update the `data_shapley_scores` accordingly.
    - Finally, **optimizer.step()** once with the sum of train+val gradients.
4. **Print Progress**: The code prints lines like:
    
    ```
    Epoch 0, Step 0, combined_loss = 4.8586
    ...
    ```
    
    showing the combined_loss progress over steps.
5. **After Training**: We have a `data_shapley_scores` array of size 60,000 storing the approximate contribution of each training sample to the single validation example’s loss.

## Outputs

- **Loss Logs**:
    
    ```
    Epoch 0, Step 0,   combined_loss = 4.8586
    Epoch 0, Step 100, combined_loss = 4.6249
    ...
    ```
    
    This indicates the model is learning as the combined loss generally decreases.
    
- **Shapley Scores**:
    
    - A 1D tensor with 60,000 entries (for MNIST).
    - Negative (large negative) scores mean **helpful** samples (their gradients align with validation).
    - Positive (large positive) scores mean **harmful** samples (their gradients conflict with validation).

You can sort and inspect these scores to find the top helpful and harmful training samples. By default, it should train for 1 epoch and print logs. You can adjust hyperparameters, number of epochs, or any other config in the script as needed.

## Sorting and Inspecting Shapley Scores

After training, we do something like:

```python
# data_shapley_scores is size [60000]

# Sort in ascending order
sorted_indices = torch.argsort(data_shapley_scores)

# Top 10 negative -> "most helpful"
print("=== Most Helpful (Negative Scores) ===")
for i in range(10):
    idx = sorted_indices[i].item()
    print(f"Sample {idx}, Score = {data_shapley_scores[idx].item():.4f}")

# Bottom 10 from the end -> "most harmful"
print("\n=== Most Harmful (Positive Scores) ===")
for i in range(10):
    idx = sorted_indices[-(i+1)].item()
    print(f"Sample {idx}, Score = {data_shapley_scores[idx].item():.4f}")
```

Then you can visualize those MNIST samples if desired.

## Interpretation

- **Negative Score**: The training sample’s gradient consistently **aligns** with the validation sample’s gradient, reducing that validation sample’s loss when included in training.
- **Positive Score**: The sample’s gradient **conflicts** with the validation sample’s gradient, tending to increase that validation sample’s loss.

### Use Cases

- **Data Cleaning**: Check if top “harmful” samples are mislabeled.
- **Data Curation**: Potentially remove or reweight harmful samples or highlight them for further inspection.
- **Further Analysis**: Repeat for multiple validation examples or the entire validation set to see overall data influence.

---

## Additional Analysis: Classical (Feature-Level) Shapley Values

In addition to **In-Run data Shapley** (which measures the contribution of training samples on the go), one can also compute **classical Shapley values** at the **feature** level to understand how different **pixels** in an MNIST image contribute to the model’s prediction. This type of analysis helps us see which parts of the digit image have a **positive** or **negative** effect on predicting the **correct digit**.

### Example Visualization

- **Objective**: For a single MNIST image (e.g., digit “7”), we want to see which pixels most strongly support (positive contribution) or oppose (negative contribution) the predicted label.

- **Interpretation**:
    
    - **Positive Shapley values** indicate pixels that push the model **toward** predicting the correct digit. In a visualization, these might be highlighted in one color (e.g., red) or shown as intensities over the digit’s important strokes.
    - **Negative Shapley values** indicate pixels that push the model **away** from the correct digit. These might be highlighted in a contrasting color (e.g., blue).

Below is a simplified illustration of how one might visualize these **classical Shapley** explanations on a digit:

![image_2025-01-04_23-18-38.png](https://github.com/parthshr370/Data-Shapley-in-One-Training-Run-Code/blob/main/image_2025-01-04_23-18-38.png)


1. **Bright/Red Regions** → strongly support the prediction of “7.”
2. **Dark/Blue Regions** → if present, might reduce the model’s confidence in predicting “7.”
3. **Neutral (Gray)** → pixels that do not contribute significantly one way or the other.

By overlaying this Shapley map onto the original digit, we can see exactly which strokes or areas of the digit matter most to the model. This complements the **in-run data Shapley** approach (which focuses on the _training samples_) by providing a feature-level perspective on **why** the model classifies a particular image the way it does.
