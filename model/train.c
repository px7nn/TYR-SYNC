// compile from the root
// gcc .\model\train.c -lm -O3

#define MLP_IMPLEMENTATION
#include "MLP.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Loading datasets...\n");
    Dataset train_d = MLP_LoadCSV("dataset/train_preprocessed.csv", 20000, 29, 1, false);
    Dataset test_d = MLP_LoadCSV("dataset/test_preprocessed.csv", 5000, 29, 1, false);

    if (train_d.n_samples == 0 || test_d.n_samples == 0) {
        printf("Failed to load dataset. Error: %s\n", MLP_ErrorString(MLP_GetLastError()));
        return 1;
    }

    printf("Loaded datasets: train %zu samples, test %zu samples\n", train_d.n_samples, test_d.n_samples);

    size_t topology[] = { 29, 32, 16, 1 };
    Activation activations[] = { ACT_RELU, ACT_RELU, ACT_SIGMOID }; 
    
    NetworkConfig cfg = {
        .topology = topology,
        .topology_size = 4,
        .activations = activations,
        .initializers = MLP_AUTO_INITIALIZERS,
        .loss = LOSS_MSE
    };

    Network net = MLP_Create_Network(&cfg);

    TrainOptions opt = MLP_DefaultTrainOptions();
    opt.max_epochs = 150;
    opt.batch_size = 32;
    opt.learning_rate = 0.005;
    opt.verbose = true;
    opt.loss_file = "graphs/loss.csv";

    printf("Starting training...\n");
    bool train_ok = MLP_Train(&net, &train_d, &opt);
    if (!train_ok) {
        printf("Training failed: %s\n", MLP_ErrorString(MLP_GetLastError()));
        return 1;
    }
    printf("Training complete.\n");

    double *preds = (double*)malloc(test_d.n_samples * sizeof(double));
    if (preds) {
        MLP_Predict_Dataset(&net, &test_d, preds);
        
        double test_loss = 0;
        for(size_t i=0; i<test_d.n_samples; i++) {
            double diff = preds[i] - test_d.outputs[i];
            test_loss += diff * diff;
        }
        test_loss /= test_d.n_samples;
        printf("Test MSE: %.6f\n", test_loss);
        free(preds);
    }
    
    MLP_Save_Network(&net, "model/tyre_model.mlp");

    MLP_Destroy_Network(&net);
    MLP_Destroy_Dataset(&train_d);
    MLP_Destroy_Dataset(&test_d);

    return 0;
}
