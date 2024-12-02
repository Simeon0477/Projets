#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cmath>

#include "../header/struct.h"
#include "../header/neuron.h"

using namespace std;

//Initialisation d'une couche
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
    layer.nombre_poids = nbr_poids;
}

//Liberation de la mémoire d'une couche
void release_layer(Couche &layer, int nbr_neurones){
    for (int i = 0; i < nbr_neurones; ++i) {
        delete layer.neurones[i].poids;
    }

    delete[] layer.neurones;
}

//initialisation du réseau de neurones
void init_NN(RN &network, int *nbr_neurones, int nbr_couche){
    network.nombre_couches = nbr_couche;

    init_layer(network.couches[i], nbr_neurones[i], nbr_neurones[i]);
    for(int i=1; i<nbr_couche; i++){
        init_layer(network.couches[i], nbr_neurones[i], nbr_neurones[i-1]);
    }
}

//Affichage de tous les poids
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

//Fonction sigmoïde
float sigmoid(float x){
    return 1.0 / (1.0 + std::exp(-x));
}

//Propagation avant
void forwardpropagation(RN &network, int *inputs){
    float agregation = 0.0f;

    //Sortie de la première couche ou couche d'entrée
    for(int i=0; i< network.couches[0].nombre_neurones; i++){
        network.couches[0].neurones[i].sortie = inputs[i];
    }

    //Propogation sur les couche restantes
    for(int i=1; i< network.nombre_couches; i++){
        for(int j=0; j< network.couches[i].nombre_neurones; j++){
            for(int k=0; k<network.couches[i-1].nombre_neurones; k++){
                agregation += network.couches[i-1].neurones[k].sortie*network.couches[i].neurones[j].poids[k];
            }

            agregation += network.couches[i].neurones[j].biais;

            network.couches[i].neurones[j].sortie = sigmoid(agregation);
        }
    }
}

//Erreur quadratique moyenne
float MSE(int *real, int *predicted, int N){
    float mse = 0.0f;

    for(int i=0; i < N; i++){
        float error = real[i] - predicted[i];
        mse += error * error;
    }

    return mse / N;
}

//Retropropagation
void backpropagation(RN &network, int *real, float learning_rate){
    //calcul de l'érreur pour la couche de sortie
    for(int i=0; i < network.couche[network.nombre_couches - 1].nombre_neurones; n++){
        
    }
}