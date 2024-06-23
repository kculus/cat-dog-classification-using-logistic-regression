import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report



def images(folder,label,image_size=(64,64)):
    labels = []
    image = []

    for filename in os.listdir(folder):
        
        img_path = os.path.join(folder,filename)
        try:
            
            img = Image.open(img_path)
            img = img.resize(image_size)
            img = np.array(img)
            if img.shape == (64,64,3):
                image.append(img)
                labels.append(label)
        
        except (IOError, OSError, Image.DecompressionBombError) as e:
            print(f"Could not open image {img_path}: {e}")

    return image,labels



dog_images,dog_labels = images("Dog",1)
cat_images,cat_labels = images("Cat",0)

image = np.array(dog_images + cat_images)
labels = np.array(dog_labels + cat_labels)

images_flattened = image.reshape(image.shape[0], -1)

images_normalized = images_flattened / 255.0

X_train, X_test, y_train, y_test = train_test_split(images_normalized, labels, test_size=0.3, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_parameters(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    
    # Forward Propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    
    # Backward Propagation
    dw = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum(A - Y)
    
    cost = np.squeeze(cost)
    grads = {"dw": dw, "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate):
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}")
    
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
    w, b = initialize_parameters(X_train.shape[0])
    
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    
    print(f"Train accuracy: {100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100}%")
    print(f"Test accuracy: {100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100}%")
    
    return parameters

Y_train = y_train.reshape(1, -1)
Y_test = y_test.reshape(1, -1)


parameters = model(X_train.T, Y_train, X_test.T, Y_test, num_iterations=300, learning_rate=0.005)


w = parameters["w"]
b = parameters["b"]
y_pred = predict(w, b, X_test.T)


accuracy = accuracy_score(Y_test.flatten(), y_pred.flatten())
print(f'Accuracy: {accuracy}')


conf_matrix = confusion_matrix(Y_test.flatten(), y_pred.flatten())
print(conf_matrix)


class_report = classification_report(Y_test.flatten(), y_pred.flatten())
print(class_report)


sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#testing image
def load_and_preprocess_image(image_path, image_size=(64, 64)):
    try:
        img = Image.open(image_path)
        img = img.resize(image_size)
        img = np.array(img)
        if img.shape == (64, 64, 3): 
            img_flattened = img.reshape(1, -1) 
            img_normalized = img_flattened / 255.0  
            return img_normalized
        else:
            raise ValueError("Image is not in the correct shape or format.")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
    


test_image_path = "kedi1.jpg"
test_image = load_and_preprocess_image(test_image_path)

y_pred = predict(parameters['w'], parameters['b'], test_image.T)
if y_pred[0][0] == 1:
    print("The image is classified as a dog.")
else:
    print("The image is classified as a cat.")
