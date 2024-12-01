#ifndef __NEURON_H__
#define __NEURON_H__

#include "struct.h"

Couche init_layer(int nbr_neurones, int nbr_poids);
void release_layer(Neuron **neurone, int nbr_neurones);
void init_NN(int *nbr_neurones, int nbr_couche);

#endif 