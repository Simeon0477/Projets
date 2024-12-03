#include "../header/struct.h"
#include "../header/neuron.h"

using namespace std;

void train(){
    //Données d'entrée pour le problème XOR
    int X[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    int y[4] = { 0, 1, 1, 0 };

    //Création du réseau de neurone
    RN network;
    int nb_neurones[] = {2, 2, 1};
    init_NN(network, nb_neurones, 3);

    float error = 1.0f;
    float outputs[4];
    while(error > 0.242204){
        //Propagation avant
        for(int i=0; i <4; i++){
            forwardpropagation(&network, X[i]);
            outputs[i] = network.couches[2].neurones[0].sortie;
        }

        //Retropropagation
        for(int i=0; i <4; i++){
            backpropagation(&network, y[i], 0.1);
        }

        //Calcul de l'erreur
        error = MSE(y, outputs, 4);

        cout << "la MSE est de : " << error <<endl;
    }
}

int main(int argc, char**args){
    train();
    return 0;
}