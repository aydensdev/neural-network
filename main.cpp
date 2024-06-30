#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>

#include "mnist.h"
#include "network.h"

#define LOG_TRAINING false

using namespace std;
using namespace chrono;

double learnRateFunction(int x)
{
    //return max(0.0, max(7-0.1*x, 6.6-0.03*x));
    //return max(0.0, 2.0 - 0.03*x);
    return max(0.0, 7.0 - 0.1*x);
    //return 1.0 / (0.1*(x+5)) + 1.5;
}

int main(int, char**)
{
    const int
        WIDTH = 2, HEIGHT = 64, INPUTH = 28*28, OUTPUTH = 10, 
        ROUNDS = 30, BATCH_SIZE = 100, DISPLAYED_INDEX = 5;

    // Load dataset

    auto loadingStart = high_resolution_clock::now();

    const string DPATH = "/home/ayden/Documents/main/machines/digits/";
    vector<vector<double>> images = read_mnist_images(DPATH+"train-images.idx3-ubyte");

    vector<int> labels = read_mnist_labels(DPATH+"train-labels.idx1-ubyte");
    vector<vector<double>> expected(labels.size(), vector<double>(10, 0.0));
    for (int i=0; i < expected.size(); i++) expected[i][labels[i]] = 1.0;

    auto loaded = high_resolution_clock::now();
    cout << "[" << duration_cast<milliseconds>(loaded-loadingStart).count();
    cout << "ms] Finished loading MNIST dataset.\n";

    // Initialize network

    Network nw = Network(WIDTH, HEIGHT, OUTPUTH, INPUTH);
    nw.SetTrainingData(images, expected);

    auto initialized = high_resolution_clock::now();
    cout << "[" << duration_cast<milliseconds>(initialized-loaded).count();
    cout << "ms] Initialized network and transported dataset.\n\n";

    printDigit(DISPLAYED_INDEX, images, labels); cout << endl;

    int total = expected.size(), full, s, barsize = 20;
    double accuracyBefore = nw.ClassificationAccuracy(labels, false);
    double learn, IV = 100.0/ROUNDS, JV = IV/total, p;

    auto trainingStart = high_resolution_clock::now();
    vector<double> trainingData;

    fstream logFile;
    if (LOG_TRAINING) logFile.open("../training_log.csv", ios::out);

    for (int i = 0; i < ROUNDS; i++)
    {
        learn = learnRateFunction(i);

        for (int j = 0; j < total; j += BATCH_SIZE)
        {
            nw.TrainingStep(j, BATCH_SIZE, learn);
            
            s = duration_cast<seconds>(system_clock::now()-trainingStart).count();
            p = i*IV+j*JV; full = round(p/100.0*barsize);

            if (p!=0)
            {
                cout << "\r\033[F\033[F[" << string(full, '=') 
                    << string(barsize-full, ' ') << "] " << round(p) << "%   \n\n"
                    << round(s/p*(100.0-p)) << "s remaining" << string(s%4, '.') << "   ";
                flush(cout);
            }
        }

        if (LOG_TRAINING) logFile << nw.ClassificationAccuracy(labels, false) << "\n";
    }

    cout << "\n[" << duration_cast<milliseconds>(system_clock::now()-trainingStart).count();
    cout << "ms] Accuracy shifted " << accuracyBefore;
    cout << "% -> " << nw.ClassificationAccuracy(labels, false) << "%\n\n";

    logFile.close();
}
