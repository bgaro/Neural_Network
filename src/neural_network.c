#include "neural_network.h"
#include "csv_to_array.h"

void get_data(FILE* train_value_file, FILE* train_labels_file, float**input_array, float **true_labels)
{
    input_array = csv_to_array_vectors(train_value_file);
    true_labels = csv_to_array_labels(train_labels_file);
}

/**
 * @brief Applies forward propagation 
 * @param SET_SIZE Dataset size (training set size)
 * @param SIZE_PREVIOUS Number of neurons is previous layer
 * @param LAYER_SIZE Number of neurons is this layer
 * @param previous_output Output of previous layer
 * @param output_matrix Output matrix for this layer (before bias & activativation)
 * @param weight_matrix Weights for this layer
 * @param bias_matrix Bias for this layer
 * @param activation_matrix store result after activation
 * @param is_output_layer if 0, use softmax, else use relu
 * @return void
 */
void feedforward(int SET_SIZE, int SIZE_PREVIOUS, int LAYER_SIZE,
                float previous_output[SET_SIZE][SIZE_PREVIOUS],
                float output_matrix[SET_SIZE][LAYER_SIZE],
                float weight_matrix[LAYER_SIZE][SIZE_PREVIOUS],
                float bias_matrix[LAYER_SIZE],
                float activation_matrix[SET_SIZE][LAYER_SIZE],
                int is_output_layer)
{
    //Multiply previous output by weight, store it in output
   
    //Add the bias
    
    //Activation
    if(is_output_layer == 0)
    {//Apply softmax
    printf("Hidden Layer activation\n")
    }else{
        printf("Last layer activation\n")
    //Apply reLU
    }
}

void reLU(int ROWS, int COLS, float input_matrix[ROWS][COLS])
{
    int i, j;
    for(i=0; i<ROWS; i++)
    {
        for(j=0; j<COLS; j++)
        {
            input_matrix[i][j] = (input_matrix[i][j] >= 0) ? input_matrix[i][j] : 0;
        }
    }
}

void softmax(int ROWS, int COLS, float input_matrix[ROWS][COLS])
{
    float sum = 0
    int i, j;
    for(i=0; i<ROWS; i++)
    {
        for(j=0; j<COLS; j++)
        {
            sum += exp(input_matrix[i][j]);
        }
    }
        for(i=0; i<ROWS; i++)
    {
        for(j=0; j<COLS; j++)
        {
            input_matrix[i][j] = exp(input_matrix[i][j])/sum;
        }
    }
}