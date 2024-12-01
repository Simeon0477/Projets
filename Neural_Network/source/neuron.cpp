#include <cstdlib>
#include <ctime>

#include "../header/struct.h"
#include "../header/neuron.h"

Couche init_layer(int nbr_neurones, int nbr_poids){
    srand((time(0)));

    Neuron** neurone = new Neuron*[nbr_neurones];

    for(int i=0; i < nbr_neurones; i++){
        neurone[i] -> poids = new float[nbr_poids];

        for(int j=0; j < nbr_poids; j++){
            neurone[i] -> poids[j] = rand() * 2.0f - 1.0f;
        }

        neurone[i] -> biais = rand() * 2.0f - 1.0f;
        neurone[i] -> sortie = 0.0f;
    }

    release_layer(neurone, nbr_neurones);

    Couche layer;
    layer.nombre_neurones = nbr_neurones;
    layer.neurones = *neurone;

    return layer;
}

void release_layer(Neuron **neurone, int nbr_neurones){
    for (int i = 0; i < nbr_neurones; ++i) {
        delete neurone[i];
    }

    delete[] neurone;
}

void init_NN(int *nbr_neurones, int nbr_couche){
    RN network;
    network.nombre_couches = nbr_couche;

    for(int i=0; i<nbr_couche; i++){
        network.couches[i] = init_layer(nbr_neurones[i], 1);
    }
}