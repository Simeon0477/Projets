#ifndef __NEURON_H__
#define __NEURON_H__

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cmath>

#include "struct.h"

void init_layer(Couche &layer,int nbr_neurones, int nbr_poids);
void release_layer(Couche &layer);
void release_network(RN &network);
void init_NN(RN &network, int *nbr_neurones, int nbr_couche);
float sigmoid(float x);
void forwardpropagation(RN &network, int *inputs);
float MSE(int *real, float *predicted, int N);
void backpropagation(RN &network, int real, float learning_rate);

#endif 