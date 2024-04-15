#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct neuron {
    double *weight;
    double bias;
    double output;
} Neuron;

typedef struct layer {
    Neuron *neurons;
    int n_neurons;
} Layer;

typedef struct neural_net {
    Layer input_layer;
    Layer hidden_layer;
    Layer output_layer;
} NeuralNetwork;

double relu(double x) {
    if (x <= 0) return 0;
    return x;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void init_neurons(Neuron *neuron, int n_weights) {
    if (neuron == NULL) exit(EXIT_FAILURE);

    neuron->weight = (double*)malloc(sizeof(double) * n_weights);
    if (neuron == NULL) exit(EXIT_FAILURE);

    for (int i = 0; i < n_weights; i++) {
        neuron->weight[i] = ((double)rand())/((double)RAND_MAX);
    }

    neuron->bias = 1;
}

void create_layer(Layer *layer, int n_neurons, int n_weights) {
    layer->neurons = (Neuron*)malloc(sizeof(Neuron) * n_neurons);
    if (layer->neurons == NULL) exit(EXIT_FAILURE);

    layer->n_neurons = n_neurons;
    for (int i = 0; i < n_neurons; i++) {
        init_neurons(&layer->neurons[i], n_weights);
    }
}

void init_network(NeuralNetwork *net, int n_input, int n_hidden, int n_output) {
    create_layer(&net->input_layer, n_input, 0);
    create_layer(&net->hidden_layer, n_hidden, n_input);
    create_layer(&net->output_layer, n_output, n_hidden);
}

void forward_propagation(NeuralNetwork *net, double *entrada) {
    for (int i = 0; i < net->hidden_layer.n_neurons; i++) {
        net->hidden_layer.neurons[i].output = net->hidden_layer.neurons[i].bias;
        for (int j = 0; j < net->input_layer.n_neurons; j++) {
            net->hidden_layer.neurons[i].output += entrada[j] * net->hidden_layer.neurons[i].weight[j];
        }

        net->hidden_layer.neurons[i].output = relu(net->hidden_layer.neurons[i].output);
    }

    for (int i = 0; i < net->output_layer.n_neurons; i++) {
        net->output_layer.neurons[i].output = net->output_layer.neurons[i].bias;
        for (int j = 0; j < net->hidden_layer.n_neurons; j++) {
            net->output_layer.neurons[i].output += net->hidden_layer.neurons[j].output * net->output_layer.neurons[i].weight[j];
        }
        
        net->output_layer.neurons[i].output = sigmoid(net->output_layer.neurons[i].output);
    }
}

void back_propagation(NeuralNetwork *net, double *entrada, double saida_esperada) {
    double learning_rate = 0.1;

    // calculate error
    double erro = net->output_layer.neurons[0].output - saida_esperada;

    // output layer backpropagation
    for (int i = 0; i < net->output_layer.n_neurons; i++) {
        Neuron *neuron = &net->output_layer.neurons[i];

        // derivative sigmoid function
        double delta = erro * (neuron->output * (1 - neuron->output));

        // ajusts weight and bias for output layer
        for (int j = 0; j < net->hidden_layer.n_neurons; j++) {
            neuron->weight[j] -= learning_rate * delta * net->hidden_layer.neurons[j].output;
        }
        neuron->bias -= learning_rate * delta;
    }

    // hidden layer backpropagation
    for (int i = 0; i < net->hidden_layer.n_neurons; i++) {
        Neuron *neuron = &net->hidden_layer.neurons[i];

        double delta = 0;
        for (int j = 0; j < net->output_layer.n_neurons; j++) {
            delta += net->output_layer.neurons[j].weight[i] * erro;
        }
        delta *= (neuron->output > 0) ? 1 : 0; // derivative relu function
        
        for (int j = 0; j < net->input_layer.n_neurons; j++) {
            neuron->weight[j] -= learning_rate * delta * entrada[j];
        }
        neuron->bias -= learning_rate * delta;
    }
}

double calculate_loss(NeuralNetwork *net, double saida_esperada) {
    double output = net->output_layer.neurons[0].output;
    return pow(output - saida_esperada, 2);
}

void train(NeuralNetwork *net, double entradas[][2], double *saidas, int n_samples, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < n_samples; i++) {
            forward_propagation(net, entradas[i]);
            total_loss += calculate_loss(net, saidas[i]);
            back_propagation(net, entradas[i], saidas[i]);
        }

        total_loss /= n_samples;

        printf("Epoch %d, Loss: %f\n", epoch, total_loss);
    }
}

double predict(NeuralNetwork *net, double *entrada) {
    forward_propagation(net, entrada);
    return net->output_layer.neurons[0].output;
}

int main(void) {
    NeuralNetwork nn;
    init_network(&nn, 2, 4, 1);

    double entradas[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double saidas[4] = {0, 1, 1, 0};
    int n_samples = 4;
    int epochs = 1000;

    train(&nn, entradas, saidas, n_samples, epochs);

    double new_input1[2] = {1, 0};
    double predict_ = predict(&nn, new_input1);
    printf("Prediction for [1, 0]: %f\n", predict_);

    double new_input2[2] = {1, 1};
    double predict2_ = predict(&nn, new_input2);
    printf("Prediction for [1, 1]: %f\n", predict2_);

    return 0;
}
