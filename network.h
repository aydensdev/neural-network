#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <functional>
using namespace std;

struct Layer
{
    Layer(int N, int prevN);

    vector<vector<double>> weights; // input weights
    vector<double> biases, activations, winputs;

    int size, SEED = unsigned(time(NULL));
    function<double()> randomWeight;
};

struct Datapoint
{
    vector<double> input, expected;
    int index;

    Datapoint(vector<double> in, vector<double> out, int i);
};

struct Network
{
    vector<Layer> layers; int W, H;
    //vector<vector<double>> inputs, expected;
    vector<Datapoint> data; // NEWT

    Network( int w, int h, int oH, int iH);

    double ActivationFunction(double x);
    double ActivationDerivative(double x);

    double CostFunction(vector<double> out, vector<double> expected);
    double CostDerivative(double out, double expected);

    vector<double> Evaluate(std::vector<double> input);  

    void SetTrainingData(vector<vector<double>> i, vector<vector<double>> o);
    void TrainingStep(int START, int BATCH_SIZE, double LEARN_RATE);
    
    double ClassificationAccuracy(vector<int> labels, bool showWrong);
};

#endif