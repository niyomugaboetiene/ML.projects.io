import numpy as np

# suppose we have 3 classes this is cateogies our model will predict
classes = ["Cat", "Dog", "Rabbit"]

# toy image just numbers not real image 
image = np.array([[1, 0],
                  [0, 1]]
                  )
# predict our model outputs this define how each model is important
# hight number more likely
# for real image classification
logits = model.predict(image)
logits = np.array([2.0, 5.0, 6.0])

# apply numercal stable softmax to avoid crash of the program
shifted_logits = logits - np.max(logits)

# exponentiate
expoential = np.exp(shifted_logits)
# probs
probs = expoential / np.sum(expoential)
print("Probabilities", probs)

# pick predicted class
# np.argmax return the index of maximum probability
prexicted_index = np.argmax(probs)
# convert the index into human readable like index 2 -> rabits
predicted_class = classes[prexicted_index]
print("Predicted class", predicted_class)