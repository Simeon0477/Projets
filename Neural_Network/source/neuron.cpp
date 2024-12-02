#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cmath>

#include "../header/struct.h"
#include "../header/neuron.h"

using namespace std;

void init_layer(Couche &layer,int nbr_neurones, int nbr_poids){
    srand((time(0)));

    layer.neurones = new Neuron[nbr_neurones];

    for(int i=0; i < nbr_neurones; i++){
        layer.neurones[i].poids = new float[nbr_poids];

        for(int j=0; j < nbr_poids; j++){
            layer.neurones[i].poids[j] = (rand()) / RAND_MAX * 2.0f - 1.0f;
        }

        layer.neurones[i].biais = (rand()) / RAND_MAX * 2.0f - 1.0f;
        layer.neurones[i].sortie = 0.0f;
    }

    layer.nombre_neurones = nbr_neurones;
}

void release_layer(Couche &layer, int nbr_neurones){
    for (int i = 0; i < nbr_neurones; ++i) {
        delete layer.neurones[i].poids;
    }

    delete[] layer.neurones;
}

void init_NN(RN &network, int *nbr_neurones, int nbr_couche, int inputsize){
    network.nombre_couches = nbr_couche;

    init_layer(network.couches[0], nbr_neurones[0], inputsize);
    for(int i=1; i<nbr_couche; i++){
        init_layer(network.couches[i], nbr_neurones[i], nbr_neurones[i-1]);
    }
}

void afficher(RN network){
    for(int i=0; i<network.nombre_couches; i++){
        for(int j=0; i<network.couches[i].nombre_neurones; j++){
            for(int k=0; k<network.couches[i].nombre_poids; k++){
                cout << "Poids no" << k << " = " << network.couches[i].neurones[j].poids[k] << endl;
            }
            cout << "biais : " << network.couches[i].neurones[j].biais << endl;
        }
    }
}

float sigmoid(float x){
    return 1.0 / (1.0 + std::exp(-x));
}

void forwardpropagation(RN network, int *inputs, int inputsize){
    float agregation = 0;

    for(int i=0; i< network.couches[0].nombre_neurones; i++){
        for(int j=0; j<inputsize;j++){
            agregation += inputs[j]*network.couches[0].neurones[i].poids[j];
        }
        agregation += network.couches[0].neurones[i].biais;

        network.couches[0].neurones[i].sortie = sigmoid(agregation);
    }

    for(i=1; i< network.nombre_couches; i++){
        for(int j=0; i< network.couches[0].nombre_neurones; j++){
            
            
            agregation += network.couches[i].neurones[j].biais;

            network.couches[i].neurones[j].sortie = sigmoid(agregation);
        }
    }
}