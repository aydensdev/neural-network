// Implementation of neural networks, based on:
// https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <execution>
#include "network.h"

using namespace std;

Layer::Layer(int N, int prevN)
: 
    biases(N, 0.0),
    activations(N, 0.0),
    winputs(N, 0.0), 
    weights(N, vector<double>(prevN))
{
    // Weight initialization

    default_random_engine gen(SEED);
    uniform_real_distribution<double> d(-8.0, 8.0);
    randomWeight = bind(d, gen);

    size = N;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < prevN; j++)
        {
            weights[i][j] = randomWeight() / double(prevN);
        }
    }
}

Datapoint::Datapoint(vector<double> in, vector<double> out, int i)
{
    input = in;
    expected = out;
    index = i;
};

Network::Network( int w, int h, int oH, int iH)
{
    W = w, H = h;
    layers.push_back(Layer(h, iH)); // first layer (recieving inputs)
    for (int i=1; i<w; i++) layers.push_back(Layer(h, h));
    layers.push_back(Layer(oH, h)); // output layer
}

double Network::ActivationFunction(double x)
{
    return 1/(1+exp(-x));
    //return max(0.1*x, x);
}

double Network::ActivationDerivative(double x)
{
    double a = ActivationFunction(x);
    return a * (1 - a);
    //return x >= 0 ? 1 : 0;
}

// Mean sqared error

double Network::CostFunction(vector<double> out, vector<double> expected)
{
    double cost = 0.0;

    for (int i = 0; i < out.size(); i++)
    {
        cost += pow(out[i]-expected[i], 2);
    }

    return cost * 0.5;
}

double Network::CostDerivative(double out, double expected)
{
    return out - expected;
}

vector<double> Network::Evaluate(vector<double> input)
{
    vector<double> layerIn = input;

    for (int l = 0; l <= W; l++) // layers
    {
        int iSize = layerIn.size();

        for (int n = 0; n < layers[l].size; n++) // neurons
        {
            layers[l].winputs[n] = layers[l].biases[n];

            for (int i = 0; i < iSize; i++) // inputs
            {
                layers[l].winputs[n] += layerIn[i] * layers[l].weights[n][i];
            }

            layers[l].activations[n] = ActivationFunction(layers[l].winputs[n]);
        }

        layerIn = layers[l].activations; // input to next layer
    }

    return layers[W].activations;
}

void Network::SetTrainingData(vector<vector<double>> i, vector<vector<double>> o)
{
    for (int dp = 0; dp < i.size(); dp++)
    {
        data.push_back(Datapoint(i[dp], o[dp], dp));
    }
}

void Network::TrainingStep(int START, int BATCH_SIZE, double LEARN_RATE)
{
    // each neuron can change its own weights only-
    // while respecting what other data points want

    double influence = LEARN_RATE / double(BATCH_SIZE);

    vector<Layer> newLayers = layers;
    
    //mutex m;

    // For each data point in mini batch

    for_each
    (
        execution::seq, 
        data.begin() + START, 
        data.begin() + START + BATCH_SIZE, 
        //[this, &newLayers, &influence, &m](auto&& dp)
        [this, &newLayers, &influence](auto&& dp)
    {
        //lock_guard<mutex> lock{m};
        double costDerivative, activationDerivative, activationNudge;
        
        // Run an evaluation and get data to critique
        // Compute the desired nudges for the output layer

        vector<double> desiredChanges, out = Evaluate(dp.input);
        for (int i=0; i < layers[W].size; i++)
        {
            costDerivative = CostDerivative(layers[W].activations[i], dp.expected[i]);
            activationDerivative = ActivationDerivative(layers[W].winputs[i]);
            desiredChanges.push_back(activationDerivative * -costDerivative);
        }

        // Backpropogate through each of the hidden layers

        for (int l=W; l > 0; l--)
        {
            int iHeight = layers[l-1].size; // height of previous layer
            vector<double> nextDesiredChanges(iHeight, 0.0);

            for (int i=0; i < layers[l].size; i++)
            {
                for (int inputNum=0; inputNum<iHeight; inputNum++)
                {
                    // nudge weights in proportion to their inputted activation's alignment to the desired change
                    newLayers[l].weights[i][inputNum] += layers[l-1].activations[inputNum] * desiredChanges[i] * influence;

                    // what does this neuron think should happen to its activations
                    activationNudge = layers[l].weights[i][inputNum] * desiredChanges[i];
                    activationDerivative = ActivationDerivative(layers[l-1].winputs[inputNum]);
                    nextDesiredChanges[inputNum] += activationNudge * activationDerivative;
                }

                // nudge biases in order to meet desired changes
                newLayers[l].biases[i] += desiredChanges[i] * influence;
            }

            // use this layer's opinion on the nudges for the previous layer
            // to repeat these calculations for the previous layer

            desiredChanges = nextDesiredChanges;
        }

        // Only adjust weights and biases for neurons reading input
        // we don't care what the network thinks should happen to input because we can't change that

        for (int n=0; n < layers[0].size; n++)
        {
            for (int inputNum=0; inputNum < dp.input.size(); inputNum++)
            {
                newLayers[0].weights[n][inputNum] += dp.input[inputNum] * desiredChanges[n] * influence;
            }

            newLayers[0].biases[n] += desiredChanges[n] * influence;
        }
    });
    

    // Apply the changes agreed upon by this batch

    layers = newLayers;
    //cout << CostFunction(Evaluate(data[START].input), data[START].expected) << endl;
}


double Network::ClassificationAccuracy(vector<int> labels, bool showWrong)
{
    double accuracy = 0.0, highscore;
    vector<double> eval; int guess;

    for (int i = 0; i < data.size(); i++)
    {
        eval = Evaluate(data[i].input);
        highscore = -99.0;
        guess = -1;

        for (int i = 0; i < eval.size(); i++)
        {
            if (eval[i] > highscore)
            {
                guess = i;
                highscore = eval[i];
            }
        }
        
        if (guess == labels[i]) accuracy ++;
        //else if (showWrong) { printDigit(i, inputs, labels); cout << guess << endl; }
    }

    return 100*accuracy/labels.size();
}
