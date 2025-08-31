import numpy as np

text = ["Postitive", "Neural", "Negative"]
logits = np.array([2.0, 0.5, -1.4])
shifted_logits = logits - np.max(logits)
exponential = np.exp(shifted_logits)
probs = exponential / np.sum(exponential)
print("Probabilities", probs)

predicted_index = np.argmax(probs)
predicted_class = text[predicted_index]

print("Predicted text", predicted_class)
