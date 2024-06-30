#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>

using namespace std;

vector<vector<double>> read_mnist_images(string path);
vector<int> read_mnist_labels(string path);
void printDigit(int index, vector<vector<double>> images, vector<int> labels);

#endif