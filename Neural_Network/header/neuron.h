#ifndef __NEURON_H__
#define __NEURON_H__

#include "struct.h"

void init_layer(Couche &layer,int nbr_neurones, int nbr_poids);
void release_layer(Couche &layer, int nbr_neurones);
void init_NN(RN &network, int *nbr_neurones, int nbr_couche, int *nbr_poids);

#endif 