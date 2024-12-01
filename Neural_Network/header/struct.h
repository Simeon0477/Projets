#ifndef __STRUCT_H__
#define __STRUCT_H__

struct Neuron{
    float *poids;
    float biais;
    float sortie;
};

typedef struct Neuron Neuron;

struct Couche{
    int nombre_neurones;
    Neuron *neurones;
};

typedef struct Couche Couche;

struct ReseauDeNeurones{
    int nombre_couches;
    Couche *couches;
};

typedef struct ReseauDeNeurones RN;

#endif 