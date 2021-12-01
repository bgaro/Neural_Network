#include "matrix.h"

/*------PROPERTIES OF NETWORK ----*/
//Layers size
#define INPUT_SIZE 784 //28px*28px = num of features
#define HIDDEN_SIZE_1 80
#define HIDDEN_SIZE_2 110
#define HIDDEN_SIZE_3 90
#define HIDDEN_SIZE_4
#define OUTPUT_SIZE 10 //One neuron for each possibility
//Set size
#define TRAINING_SET_SIZE 2000
#define TEST_SET_SIZE 10000
//Parameters
#define LEARNING_RATE 0.05
#define EPOCH 50

/*-----------MATRIX DEFINTION----------*/
/*---------FORWARD PROPAGATION----------*/
//Output of each layer (= result after applying everything)
float output_hidden_1[TRAINING_SET_SIZE][HIDDEN_SIZE_1];
float output_hidden_2[TRAINING_SET_SIZE][HIDDEN_SIZE_2];
float output_hidden_3[TRAINING_SET_SIZE][HIDDEN_SIZE_3];
float output_hidden_4[TRAINING_SET_SIZE][HIDDEN_SIZE_4];
float output_final[TRAINING_SET_SIZE][OUTPUT_SIZE];
//Bias
float bias_hidden_1[HIDDEN_SIZE_1];
float bias_hidden_2[HIDDEN_SIZE_2];
float bias_hidden_3[HIDDEN_SIZE_3];
float bias_hidden_4[HIDDEN_SIZE_4];
float bias_output[OUTPUT_SIZE];
//Activation
float activation_hidden_1[TRAINING_SET_SIZE][HIDDEN_SIZE_1];
float activation_hidden_2[TRAINING_SET_SIZE][HIDDEN_SIZE_2];
float activation_hidden_3[TRAINING_SET_SIZE][HIDDEN_SIZE_3];
float activation_hidden_4[TRAINING_SET_SIZE][HIDDEN_SIZE_4];
//Weigths (weight_hidden_1 = weight between input and hidden1)
float weight_hidden_1[HIDDEN_SIZE_1][INPUT_SIZE];
float weight_hidden_2[HIDDEN_SIZE_2][HIDDEN_SIZE_1];
float weight_hidden_3[HIDDEN_SIZE_3][HIDDEN_SIZE_2];
float weight_hidden_4[HIDDEN_SIZE_4][HIDDEN_SIZE_3];
float weight_output[HIDDEN_SIZE_4][OUTPUT_SIZE];

/*----------BACKWARD PROPAGATION----------*/
//Output after the layer
float back_output_hidden_1[TRAINING_SET_SIZE][HIDDEN_SIZE_1];
float back_output_hidden_2[TRAINING_SET_SIZE][HIDDEN_SIZE_2];
float back_output_hidden_3[TRAINING_SET_SIZE][HIDDEN_SIZE_3];
float back_output_hidden_4[TRAINING_SET_SIZE][HIDDEN_SIZE_4];
float back_output_final[TEST_SET_SIZE][OUTPUT_SIZE];
//Bias
float back_bias_hidden_1[HIDDEN_SIZE_1];
float back_bias_hidden_2[HIDDEN_SIZE_2];
float back_bias_hidden_3[HIDDEN_SIZE_3];
float back_bias_hidden_4[HIDDEN_SIZE_4];
float back_bias_output[OUTPUT_SIZE];
//Activation
float back_activation_hidden_1[TRAINING_SET_SIZE][HIDDEN_SIZE_1];
float back_activation_hidden_2[TRAINING_SET_SIZE][HIDDEN_SIZE_2];
float back_activation_hidden_3[TRAINING_SET_SIZE][HIDDEN_SIZE_3];
float back_activation_hidden_4[TRAINING_SET_SIZE][HIDDEN_SIZE_4];
//Weights
float back_weight_hidden_1[HIDDEN_SIZE_1][INPUT_SIZE];
float back_weight_hidden_2[HIDDEN_SIZE_2][HIDDEN_SIZE_1];
float back_weight_hidden_3[HIDDEN_SIZE_3][HIDDEN_SIZE_2];
float back_weight_hidden_4[HIDDEN_SIZE_4][HIDDEN_SIZE_3];
float back_weight_output[HIDDEN_SIZE_4][OUTPUT_SIZE];

void get_data(FILE* train_value_file, FILE* train_labels_file, float**input_array, float **true_labels);

void feedforward(int SET_SIZE, int SIZE_PREVIOUS, int LAYER_SIZE,
                float previous_output[SET_SIZE][SIZE_PREVIOUS],
                float output_matrix[SET_SIZE][LAYER_SIZE],
                float weight_matrix[LAYER_SIZE][SIZE_PREVIOUS],
                float bias_matrix[LAYER_SIZE],
                float activation_matrix[SET_SIZE][LAYER_SIZE],
                int is_output_layer);