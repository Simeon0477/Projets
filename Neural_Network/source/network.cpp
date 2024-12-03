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
void release_layer(Couche &layer){
    for (int i = 0; i < layer.nombre_neurones; ++i) {
        delete layer.neurones[i].poids;
    }

    delete[] layer.neurones;
    layer.neurones = nullptr;
}

void release_network(RN &network){
    for(int i=0; i < network.nombre_couches; i++){
        release_layer(network.couches[i]);
    }
    delete[] network.couches;
    network.couches = nullptr;
}

//initialisation du réseau de neurones
void init_NN(RN &network, int *nbr_neurones, int nbr_couche){
    network.nombre_couches = nbr_couche;
    network.couches = new Couche[nbr_couche];

    init_layer(network.couches[0], nbr_neurones[0], nbr_neurones[0]);
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
    return 1.0 / (1.0 + exp(-x));
}

//Propagation avant
void forwardpropagation(RN *network, int *inputs){
    //Sortie de la première couche ou couche d'entrée
    for(int i=0; i< network->couches[0].nombre_neurones; i++){
        network->couches[0].neurones[i].sortie = inputs[i];
    }

    //Propogation sur les couche restantes
    for(int i=1; i< network->nombre_couches; i++){
        Couche* actu_layer = &network->couches[i];
        Couche* previous_layer = &network->couches[i - 1];

        for(int j=0; j< actu_layer->nombre_neurones; j++){
            Neuron* neuron = &actu_layer->neurones[j];

            //Fonction d'agragation
            float agregation = 0.0f;

            for(int k=0; k<previous_layer->nombre_neurones; k++){
                agregation += previous_layer->neurones[k].sortie*neuron->poids[k];
            }

            agregation += neuron->biais;

            //Application de la fonction d'activation
            neuron->sortie = sigmoid(agregation);
        }
    }
}

//Erreur quadratique moyenne
float MSE(int *real, float *predicted, int N){
    float mse = 0.0f;

    for(int i=0; i < N; i++){
        float error = real[i] - predicted[i];
        mse += error * error;
    }

    return mse / N;
}

//Retropropagation
void backpropagation(RN *network, int real, float learning_rate){
    int ind_sortie = network->nombre_couches - 1;

    //calcul de l'érreur pour la couche de sortie
    for(int i=0; i < network->couches[ind_sortie].nombre_neurones; i++){
        Neuron *neuron = &network->couches[ind_sortie].neurones[i];
        float sortie = neuron->sortie;

        neuron->error = (sortie - real) * sortie * (1 - sortie);
    }

    //Rétropropagation
    for(int i=ind_sortie - 1; i>=0; i--){
        Couche* actu_layer = &network->couches[i];
        Couche* next_layer = &network->couches[i + 1];

        for(int j=0; j < actu_layer->nombre_neurones; j++){
            Neuron* neuron = &actu_layer->neurones[j];
            float somme_erreurs = 0.0f;

            //Calcul de l'erreur propagée
            for(int k=0; k < next_layer->nombre_neurones; k++){
                somme_erreurs += next_layer->neurones[k].poids[j] * next_layer->neurones[k].error;
            }

            neuron->error = somme_erreurs * neuron->sortie * (1 - neuron->sortie);
        }
    }

    //Mise à jour des poids et des biais
    for(int i=1; i<network->nombre_couches; i++){
        Couche* actu_layer = &network->couches[i];
        Couche* previous_layer = &network->couches[i - 1];

        for(int j=0; j<actu_layer->nombre_neurones; j++){
            Neuron* neuron = &actu_layer->neurones[j];

            //Mettre à jour les poids
            for(int k=0; k<previous_layer->nombre_neurones; k++){
                neuron->poids[k] -= learning_rate * neuron->error * previous_layer->neurones[k].sortie;
            }

            //Mettre à jour les biais
            neuron->biais -= learning_rate * neuron->error;
        }
    }
}
