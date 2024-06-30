// Functions for reading and visualing MNIST dataset
// adapted stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c

#include <iostream>
#include <fstream>
#include <vector>
#include <functional>

using namespace std;

unsigned int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((unsigned int)c1 << 24) + ((unsigned int)c2 << 16) + ((unsigned int)c3 << 8) + c4;
}

vector<vector<double>> read_mnist_images ( string path )
{
    ifstream file(path, ios::binary);
    int magic_number = 0, rows = 0, cols = 0, num_img = 0;

    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    file.read((char *)&num_img, sizeof(num_img)), num_img = reverseInt(num_img);
    file.read((char *)&rows, sizeof(rows)), rows = reverseInt(rows);
    file.read((char *)&cols, sizeof(cols)), cols = reverseInt(cols);
    int img_size = rows * cols;

    cout << "Reading MNIST images, #" << magic_number;
    cout << "   size: " << rows << "x" << cols << endl;

    vector<vector<unsigned char>> raw_data(num_img, vector<unsigned char>(img_size));
    for (int i = 0; i < num_img; i++) 
    {
        file.read(reinterpret_cast<char*>(raw_data[i].data()), img_size);
    }

    vector<vector<double>> mnist_images(num_img, vector<double>(img_size));
    for (int i = 0; i < num_img; i++) 
    {
        for (int j = 0; j < img_size; j++)
        {
            mnist_images[i][j] = static_cast<double>(raw_data[i][j]) / 255.0;
        }
    }
    
    return mnist_images;
}

vector<int> read_mnist_labels ( string path )
{
    ifstream file(path, ios::binary);
    int magic_number = 0, num_labels = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    file.read((char *)&num_labels, sizeof(num_labels)), num_labels = reverseInt(num_labels);

    cout << "Reading MNIST labels, #" << magic_number;
    cout << " amount: " << num_labels << endl;

    vector<unsigned char> raw(num_labels);
    vector<int> dataset(num_labels);

    for(int i = 0; i < num_labels; i++) 
    {
        file.read(reinterpret_cast<char*>(&raw[i]), 1);
        dataset[i] = (int)raw[i];
    }

    return dataset;
}

void printDigit(int index, vector<vector<double>> images, vector<int> labels)
{
    double value;

    for (int i = 0; i < 28 * 28; i++) 
    {
        if (i == 0) { cout << labels[index] << ' '; }
        else
        {
            value = images[index][i];
            cout << (value > 0.94 ? '#' : (value > 0.3 ? '+' : ' '));
            cout << ((i + 1) % 28 == 0 ? '\n' : ' ');
        }
    }
}