# %%
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import random
from statistics import mean

trainingImagesRawData = './Data/train-images.idx3-ubyte'
trainingImages = idx2numpy.convert_from_file(trainingImagesRawData)

trainingLabelsRawData = './Data/train-labels.idx1-ubyte'
trainingLabels = idx2numpy.convert_from_file(trainingLabelsRawData)


# Normalise the datasets
trainingImages = trainingImages / 255.0

trainingImages = [x.flatten() for x in trainingImages] 


# %%
def displayImage(dataset, num):
    image = np.array(dataset[:,num], dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

# %%
def createDesiredOutputs(labels):
    desiredOutputs = np.zeros((10, len(labels)))
    row = 0
    for label in labels:
        
        desiredOutputs[label, row] = 1
        row += 1
    
    return desiredOutputs

# %%
desiredOutputs = createDesiredOutputs(trainingLabels)


# %%
trainingImages = np.array(trainingImages).T

# %%
testImagesRawData = './Data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
testImages = idx2numpy.convert_from_file(testImagesRawData)

testLabelsRawData = './Data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
testLabels = idx2numpy.convert_from_file(testLabelsRawData)


# Normalise the datasets
testImages = testImages / 255.0

testImages = [x.flatten() for x in testImages] 

desiredtestOutputs = createDesiredOutputs(testLabels)


testImages = np.array(testImages).T

# %%
layers = [] # The hidden layers and output layer in order

# %%
def initialiseNetwork(hiddenLayerConfig):
    
    previousLayerSize = 28*28 # Size of input layer as 28*28 pixel images
    hiddenLayerConfig.append(10) # Add in output layer as range from digits 0 to 9
    
    for layerConfig in hiddenLayerConfig:
        layerWeights = np.random.randn(layerConfig, previousLayerSize)
        layerBiases = np.random.randn(1,layerConfig)
        layers.append([layerWeights, layerBiases])
        previousLayerSize = layerConfig
        



# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidPrime(x):
    return sigmoid(x) * (1 - sigmoid(x))


# %%
def softmax(outputArr):
    return outputArr / np.sum(outputArr)

# %%
def feedforward(inputs): # inputs are going to be in the form of a matrix n * m where n is the number of input neurons and m the number of training cases
    
    unweightedOutputs = [] # this the one in the book which was like z^l
    weightedOutputs = [] # this is z after applying activation e.g sigmoid
    currentLayer = inputs
    i=1
    for layer in layers: 

        unweightedOutput = np.matmul(layer[0], currentLayer) 
        weightedOutput = sigmoid(unweightedOutput)
        
        unweightedOutputs.append(unweightedOutput)
        weightedOutputs.append(weightedOutput)

        currentLayer = weightedOutput
        
        i+= 1

        
    finalOutput = weightedOutputs[-1]
    
    return finalOutput, weightedOutputs, unweightedOutputs
    

# %%
def calculateCost(desiredOutputs, actualOutputs):
    numberOfTrainingCases = actualOutputs.shape[1]
    
    costs = []
    for trainingCase in range(numberOfTrainingCases):

        trainingCaseOuput = actualOutputs[:,trainingCase]
        trainingCaseDesired = desiredOutputs[:,trainingCase]
        # print(trainingCaseDesired, trainingCaseOuput)
        costForTrainingCase = np.sum([(x - y)**2 for x, y in zip(trainingCaseDesired,trainingCaseOuput)])/10
        costs.append(costForTrainingCase)
    return costs

# %%
def backProp(unweightedOutputs, activations, inputs, finalOuputsBatch, desiredOutputsBatch):
    allLayerErrors = []
    numberOfCases = inputs.shape[1]
    # Error in output layer 
    outputLayerErrorP1 = 2 * (finalOuputsBatch - desiredOutputsBatch) # Because this is derivative of cost function (y-x)^2 to 2(x-y)
    outputLayerErrorP2 = sigmoidPrime(unweightedOutputs[-1])
    outputLayerError = np.multiply(outputLayerErrorP1, outputLayerErrorP2) # This is schur product not matrix multiplication
    
    currentLayerError = outputLayerError
    allLayerErrors.insert(0, currentLayerError)

    for layer in range(len(layers)-1):

        lMinusOneErrorP1 = np.matmul(layers[-1 * (layer+1)][0].T, currentLayerError) 
        lMinusOneErrorP2 = sigmoidPrime(unweightedOutputs[-1 * (layer+2)])
        lMinusOneError = np.multiply(lMinusOneErrorP1, lMinusOneErrorP2)
        
        currentLayerError = lMinusOneError
        allLayerErrors.insert(0, currentLayerError)
    
    allLayerBiasesDerivatives = allLayerErrors
    allLayerWeightsDerivatives = []
    
    
    previousLayerActivations = inputs
    i = 0
    for layerError in allLayerErrors:
        layerWeightDerivative = np.matmul(previousLayerActivations, layerError.T)
        
        allLayerWeightsDerivatives.append(layerWeightDerivative)
        previousLayerActivations = activations[i]
        i += 1
        

    averageLayerBiasesDerivative = [x.mean(1) for x in allLayerBiasesDerivatives]
    averageLayerWeightsDerivative = [x/numberOfCases for x in allLayerWeightsDerivatives]
    
    
    return averageLayerBiasesDerivative, averageLayerWeightsDerivative

# %%
def createBatches(data, labels, batchSize):
    dataBatches = np.array_split(data, data.shape[1]/batchSize, axis=1)
    labelBatches = np.array_split(labels, data.shape[1]/batchSize, axis=1)
    
    
    batches = list(zip(dataBatches, labelBatches))
    
    return batches

# %%
def stochasticGradientDescent(trainingRate):
    
    batches = createBatches(trainingImages, desiredOutputs, 100)
    totalCosts = []
    
    for trainingBatch in batches:
        trainingBatchFinalOutput, trainingBatchWeightedOutputs, trainingBatchUnweightedOutputs = feedforward(trainingBatch[0])

        averageLayerBiasesDerivative, averageLayerWeightsDerivative = backProp(trainingBatchUnweightedOutputs, trainingBatchWeightedOutputs, trainingBatch[0], trainingBatchFinalOutput, trainingBatch[1])
        
        for i in range(len(layers)):
            layers[i][0] -= trainingRate * averageLayerWeightsDerivative[i].T
            layers[i][1] -= trainingRate * averageLayerBiasesDerivative[i].T
            
        # print("Weight updates:", [np.mean(np.abs(trainingRate * grad)) for grad in averageLayerWeightsDerivative])
        
        totalCosts.append(mean(calculateCost(trainingBatchFinalOutput, trainingBatch[1])))
        
    return totalCosts
        

# %%
initialiseNetwork([100,20])

trainingRate=3

for i in range(50):
    totalCosts = stochasticGradientDescent(trainingRate)
    totalAverageCost = mean(totalCosts)
    
    
    print('Epoch: ', i+1)
    print("Current Network cost: ", totalAverageCost)
    if i == 40:
        trainingRate = 0.6
    

# %%
finalOutput, weightedOutputs, unweightedOutputs = feedforward(trainingImages[:,1])

print(finalOutput)

# %%
def feedforwardTest(inputs): # inputs are going to be in the form of a matrix n * m where n is the number of input neurons and m the number of training cases
    
    unweightedOutputs = [] # this the one in the book which was like z^l
    weightedOutputs = [] # this is z after applying activation e.g sigmoid
    currentLayer = inputs
    
    for layer in layers: 

        unweightedOutput = np.matmul(layer[0], currentLayer) 
        weightedOutput = sigmoid(unweightedOutput)
        
        unweightedOutputs.append(unweightedOutput)
        weightedOutputs.append(weightedOutput)

        currentLayer = weightedOutput


        
    finalOutput = weightedOutputs[-1]

    
    finalOutput = np.apply_along_axis(softmax, axis=0, arr=finalOutput)
    return finalOutput, weightedOutputs, unweightedOutputs

# %%
def recordGuesses(arr):

    return np.argmax(arr)


def checkGuesses(guesses, answers):#
    mark = 0
    for i in range(len(guesses)):
        if guesses[i] == answers[i]:
            mark += 1
            
    return mark

# %%


batches = createBatches(testImages, desiredtestOutputs, 100)
totalCosts = []
marks = []

i = 100
for trainingBatch in batches:
    trainingBatchFinalOutput, trainingBatchWeightedOutputs, trainingBatchUnweightedOutputs = feedforwardTest(trainingBatch[0])
        
    # print("Weight updates:", [np.mean(np.abs(trainingRate * grad)) for grad in averageLayerWeightsDerivative])
    

    totalCosts.append(mean(calculateCost(trainingBatchFinalOutput, trainingBatch[1])))
    
    correctGuesses = 0
    guesses = []
    for col in range(trainingBatchFinalOutput.shape[1]):
        guess = recordGuesses(trainingBatchFinalOutput[:,col])
        guesses.append(guess)
        
    answers = testLabels[(i-100):i]
    
    marks.append(checkGuesses(guesses, answers))
    i += 100
    
print(mean(marks))
    
    
    
totalAverageCost = mean(totalCosts)

print("Current Network cost: ", totalAverageCost)


