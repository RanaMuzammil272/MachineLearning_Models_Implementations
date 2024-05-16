import random
import math

def sigmoid(x):
    return (1/(1+math.exp(-x)))

def sigmoid_derivative(x):
    return(x*(1-x))

def squared_err(expected,obtained):
    return (1/2*((expected-obtained)*2));

class NeuralNetwork:

    def __init__(self,input_size,hidden_layer,output_size):
        self.input_size=input_size
        self.hidden_layer=hidden_layer
        self.output_size=output_size
        self.hidden_layer_weights=[];
        self.hidden_layer_output_weights=[];
        self.hidden_layer_bias=[];
        self.output_layer_bias=[];
        for i in range(input_size):
            arr=[]
            for j in range(hidden_layer):
                arr.append(random.randint(0, 1))
            self.hidden_layer_weights.append(arr)
        for i in range(hidden_layer):
            arr=[]
            for j in range(output_size):
                arr.append(random.randint(0, 1))
            self.hidden_layer_output_weights.append(arr)
        for i in range(hidden_layer):
            self.hidden_layer_bias.append(random.randint(0, 1))
        for i in range(output_size):
            self.output_layer_bias.append(random.randint(0, 1))

    def forward_propogation(self,inputs):
        hidden_results=[]
        output_results=[]
        for i in range(self.hidden_layer):
            temp=0
            for j in range(self.input_size):
                temp+=inputs[j]*self.hidden_layer_weights[j][i];
            temp+=self.hidden_layer_bias[i]
            hidden_results.append(sigmoid(temp));
        for i in range(self.output_size):
            temp=0
            for j in range(self.hidden_layer):
                temp+=hidden_results[j]*self.hidden_layer_output_weights[j][i];
            temp+=self.output_layer_bias[i]
            output_results.append(sigmoid(temp));
        
        return hidden_results,output_results;

    def backpropogation(self,inputs,expected_output,learning_rate):

        hidden_results, output_results = self.forward_propogation(inputs)
        err=[]
        output_delta=[]
        for i in range(self.output_size):
            err.append(squared_err(expected_output[i],output_results[i]));
        for i in range(self.output_size):
            output_delta.append(err[i]*sigmoid_derivative(output_results[i]));
        hidden_delta=[];
        for i in range(self.hidden_layer):
            sum=0;
            for j in range(self.output_size):
                sum=sum+(sigmoid_derivative(hidden_results[i])*output_delta[j]*self.hidden_layer_output_weights[i][j]);
            hidden_delta.append(sum);
        for i in range(self.output_size):
            self.output_layer_bias[i]=self.output_layer_bias[i]+learning_rate*output_delta[i];
        for i in range(self.hidden_layer):
            self.hidden_layer_bias[i]=self.hidden_layer_bias[i]+learning_rate*hidden_delta[i];
        for i in range(self.hidden_layer):
            for j in range(self.output_size):
                self.hidden_layer_output_weights[i][j]=self.hidden_layer_output_weights[i][j]+learning_rate*output_delta[j]*hidden_results[i];
        for i in range(self.input_size):
            for j in range(self.hidden_layer):
                self.hidden_layer_weights[i][j]=self.hidden_layer_weights[i][j]+learning_rate*hidden_delta[j]*inputs[i];

    def train(self, inputs, expected_output, learning_rate, iterations):
            for iterate in range(iterations):
                for i in range(len(inputs)):
                    self.backpropogation(inputs[i], expected_output[i], learning_rate)
                
                error = sum(sum((expected_output[i][j] - self.forward_propogation(inputs[i])[1][j]) ** 2 for j in range(len(expected_output[i]))) for i in range(len(inputs))) / len(inputs)

                if iterate % 1000 == 0:
                    print(f"Iterations {iterate}: Error = {error}")


inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [[0], [1], [1], [0]]
neural_net = NeuralNetwork(2, 4, 1)
neural_net.train(inputs, targets, learning_rate=0.1, iterations=10000)
print("Test Results:")
for i in range(len(inputs)):
    input_data = inputs[i]
    target_data = targets[i]
    output = neural_net.forward_propogation(input_data)[1][0]
    print(f"Input: {input_data}, Target: {target_data}, Output: {output}")





   








    

