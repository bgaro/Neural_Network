#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "activation.h"
#include "csv_to_array.h"
#define INPUT_NEURON 784
#define HIDDEN_NEURON_1 80
#define HIDDEN_NEURON_2 110
#define HIDDEN_NEURON_3 100
#define HIDDEN_NEURON_4 90
#define OUTPUT_NEURON 10
#define LAYER_NUM 3
#define TRAINING_SET_SIZE 2000
#define OUPUT_SIZE 1
#define EPOCH 1

int main()
{

    FILE *train_vectors_stream = fopen("./data/test_vectors (1).csv", "r");
    if (train_vectors_stream == NULL)
    {
        printf("Error opening file\n");
        return 1;
    }
    FILE *train_labels_stream = fopen("./data/test_labels (1).csv", "r");
    if (train_labels_stream == NULL)
    {
        printf("Error opening file\n");
        return 1;
    }

    float **input_array;
    float learning_rate = 0.05;
    // feed forward matrix
    matrix_t *input_layer_transpose = matrix_create(1, INPUT_NEURON);
    matrix_t *input_layer = matrix_create(INPUT_NEURON, 1);
    matrix_t *hidden_layer_1 = matrix_create(HIDDEN_NEURON_1, 1);
    matrix_t *hidden_layer_2 = matrix_create(HIDDEN_NEURON_2, 1);
    matrix_t *hidden_layer_3 = matrix_create(HIDDEN_NEURON_3, 1);
    matrix_t *hidden_layer_4 = matrix_create(HIDDEN_NEURON_4, 1);
    matrix_t *output_layer = matrix_create(OUTPUT_NEURON, 1);

    matrix_t *bias_hidden_1 = matrix_create(HIDDEN_NEURON_1, 1);
    matrix_initialize_random(bias_hidden_1, 100);
    matrix_t *bias_hidden_2 = matrix_create(HIDDEN_NEURON_2, 1);
    matrix_initialize_random(bias_hidden_2, 100);
    matrix_t *bias_hidden_3 = matrix_create(HIDDEN_NEURON_3, 1);
    matrix_initialize_random(bias_hidden_3, 100);
    matrix_t *bias_hidden_4 = matrix_create(HIDDEN_NEURON_4, 1);
    matrix_initialize_random(bias_hidden_4, 100);

    matrix_t *bias_output = matrix_create(OUTPUT_NEURON, 1);
    matrix_initialize_random(bias_output, 100);

    matrix_t *weight_input_hidden_1 = matrix_create(HIDDEN_NEURON_1, INPUT_NEURON);
    matrix_initialize_random(weight_input_hidden_1, 100);
    matrix_t *weight_input_hidden_2 = matrix_create(HIDDEN_NEURON_2, HIDDEN_NEURON_1);
    matrix_initialize_random(weight_input_hidden_2, 100);
    matrix_t *weight_input_hidden_3 = matrix_create(HIDDEN_NEURON_3, HIDDEN_NEURON_2);
    matrix_initialize_random(weight_input_hidden_3, 100);
    matrix_t *weight_input_hidden_4 = matrix_create(HIDDEN_NEURON_4, HIDDEN_NEURON_3);
    matrix_initialize_random(weight_input_hidden_4, 100);
    matrix_t *weight_hidden_4_output = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON_4);
    matrix_initialize_random(weight_hidden_4_output, 100);

    matrix_t *activation_hidden_1_matrix = matrix_create(HIDDEN_NEURON_1, 1);
    matrix_t *activation_hidden_2_matrix = matrix_create(HIDDEN_NEURON_2, 1);
    matrix_t *activation_hidden_3_matrix = matrix_create(HIDDEN_NEURON_3, 1);
    matrix_t *activation_hidden_4_matrix = matrix_create(HIDDEN_NEURON_4, 1);
    matrix_t *activation_output_matrix = matrix_create(OUTPUT_NEURON, 1);

    // backpropagation matrix
    /*
        matrix_t *derivate_error_output_layer = matrix_create(OUTPUT_NEURON, 1);
        matrix_t *derivate_output = matrix_create(OUTPUT_NEURON, 1);
        matrix_t *derivate_output_diag = matrix_create(OUTPUT_NEURON, OUTPUT_NEURON);
        matrix_t *derivate_error_output_layer_diag = matrix_create(OUTPUT_NEURON, OUTPUT_NEURON);
        matrix_t *derivate_output_activiation = matrix_create(OUTPUT_NEURON, 1);

        matrix_t *derivate_error_hidden_layer = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON);
        matrix_t *derivate_error_hidden_layer_transpose = matrix_create(HIDDEN_NEURON, OUTPUT_NEURON);

        matrix_t *error_weight_gradient_hidden = matrix_create(HIDDEN_NEURON, INPUT_NEURON);
        matrix_t *error_weight_gradient_output = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON);
        matrix_t *error_weight_gradient_output_step_transpose = matrix_create(HIDDEN_NEURON, OUTPUT_NEURON);
        matrix_t *error_weight_gradient_output_step = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON);
        matrix_t *error_weight_gradient_bias_output_step = matrix_create(OUTPUT_NEURON, 1);
        matrix_t *error_weight_gradient_bias_output = matrix_create(OUTPUT_NEURON, 1);
        matrix_t *error_weight_gradient_bias_hidden = matrix_create(HIDDEN_NEURON, 1);

        matrix_t *derivate_hidden = matrix_create(HIDDEN_NEURON, 1);
        matrix_t *derivate_hidden_diag = matrix_create(HIDDEN_NEURON, HIDDEN_NEURON);
        matrix_t *derivate_hidden_error = matrix_create(HIDDEN_NEURON, 1);

        matrix_t *activation_input_matrix = matrix_create(INPUT_NEURON, 1);
        matrix_t *activation_input_matrix_transpose = matrix_create(1, INPUT_NEURON);

        matrix_t *error_weight_gradient_hidden_step = matrix_create(HIDDEN_NEURON, INPUT_NEURON);
        matrix_t *error_weight_gradient_bias_hidden_step = matrix_create(HIDDEN_NEURON, 1);

        matrix_t *derivate_error_hidden_layer_diag = matrix_create(HIDDEN_NEURON, HIDDEN_NEURON);
    */
    for (int j = 0; j < EPOCH; j++)
    {
        for (int i = 0; i < TRAINING_SET_SIZE; i++)
        {

            // Feed forward process

            // initialisation of input layer

            input_array = csv_to_array_vectors(train_vectors_stream);
            matrix_initialize(input_layer_transpose, 1, INPUT_NEURON, input_array);
            matrix_transpose(input_layer_transpose, input_layer);

            // feed forward on hidden layer 1
            matrix_multiply(weight_input_hidden_1, input_layer, hidden_layer_1);
            matrix_add(hidden_layer_1, bias_hidden_1);
            reLU(hidden_layer_1, activation_hidden_1_matrix);

            // feed forward on hidden layer 2
            matrix_multiply(weight_input_hidden_2, hidden_layer_1, hidden_layer_2);
            matrix_add(hidden_layer_2, bias_hidden_2);
            reLU(hidden_layer_2, activation_hidden_2_matrix);

            // feed forward on hidden layer 3
            matrix_multiply(weight_input_hidden_3, hidden_layer_2, hidden_layer_3);
            matrix_add(hidden_layer_3, bias_hidden_3);
            reLU(hidden_layer_3, activation_hidden_3_matrix);

            // feed forward on hidden layer 4
            matrix_multiply(weight_input_hidden_4, hidden_layer_3, hidden_layer_4);
            matrix_add(hidden_layer_4, bias_hidden_4);
            reLU(hidden_layer_4, activation_hidden_4_matrix);

            // feed forward on output layer
            matrix_multiply(weight_hidden_4_output, activation_hidden_4_matrix, output_layer);
            matrix_add(output_layer, bias_output);
            reLU(output_layer, activation_output_matrix);

            free(input_array[0]);
            free(input_array);
            /*
                        // error function gradiant

                        // yj - dkj for j in Y (output layer)
                        // matrix_initialize(expected_output_step, expected_output_step->rows, expected_output_step->cols, &expected_output->data[i][0]);
                        derivate_error_output_layer->data[0] = csv_to_array_labels(train_labels_stream) - activation_output_matrix->data[0];
                        // dEk/dyj for j in Y (output layer)

                        // dEk/dyj for j in Z \ (Y U X) (hidden layer)

                        reLU_derivate(output_layer, derivate_output);
                        matrix_diagonalize(derivate_output, derivate_output_diag);
                        matrix_multiply(derivate_output_diag, derivate_error_output_layer, derivate_output_activiation); // dEk/dyr * sigma'r(xi r) for r in j->
                        matrix_multiply(derivate_output_activiation, weight_hidden_output, derivate_error_hidden_layer); // dEk/dyj

                        matrix_transpose(derivate_error_hidden_layer, derivate_error_hidden_layer_transpose);
                        // dEk/dwij for j in Y(output layer)
                        matrix_multiply(activation_hidden_matrix, derivate_output_activiation, error_weight_gradient_output_step_transpose);
                        matrix_transpose(error_weight_gradient_output_step_transpose, error_weight_gradient_output_step); // transpose so it is of the form of weight matrix
                        matrix_add(error_weight_gradient_output, error_weight_gradient_output_step);                      // sum of each training set

                        // bias of output layer update

                        matrix_diagonalize(derivate_error_output_layer, derivate_error_output_layer_diag);
                        matrix_multiply(derivate_error_output_layer_diag, bias_output, error_weight_gradient_bias_output_step); // dek/dwij
                        matrix_add(error_weight_gradient_bias_output, error_weight_gradient_bias_output_step);

                        // dEk/dwij for j in Z \ (Y U X) (hidden layer)
                        reLU_derivate(hidden_layer, derivate_hidden);
                        matrix_diagonalize(derivate_hidden, derivate_hidden_diag);
                        matrix_multiply(derivate_hidden_diag, derivate_error_hidden_layer_transpose, derivate_hidden_error);

                        reLU(input_layer, activation_input_matrix);
                        matrix_transpose(activation_input_matrix, activation_input_matrix_transpose);
                        matrix_multiply(derivate_hidden_error, activation_input_matrix_transpose, error_weight_gradient_hidden_step);
                        matrix_add(error_weight_gradient_hidden, error_weight_gradient_hidden_step);

                        // bias of hidden layer update
                        matrix_diagonalize(derivate_error_hidden_layer_transpose, derivate_error_hidden_layer_diag);
                        matrix_multiply(derivate_error_hidden_layer_diag, bias_hidden, error_weight_gradient_bias_hidden_step);
                        matrix_add(error_weight_gradient_bias_hidden, error_weight_gradient_bias_hidden_step);

                        matrix_print(output_layer);
                        // free memory

                    }
                    // update weight
                    matrix_multiply_constant(error_weight_gradient_bias_output, -(1.0 / TRAINING_SET_SIZE) * learning_rate);
                    matrix_multiply_constant(error_weight_gradient_bias_hidden, -(1.0 / TRAINING_SET_SIZE) * learning_rate);
                    matrix_multiply_constant(error_weight_gradient_output, -(1.0 / TRAINING_SET_SIZE) * learning_rate);
                    matrix_multiply_constant(error_weight_gradient_hidden, -(1.0 / TRAINING_SET_SIZE) * learning_rate);
                    matrix_add(bias_output, error_weight_gradient_bias_output);
                    matrix_add(bias_hidden, error_weight_gradient_bias_hidden);
                    matrix_add(weight_hidden_output, error_weight_gradient_output);
                    matrix_add(weight_input_hidden, error_weight_gradient_hidden);

                    // reset file pointer
                    fseek(train_vectors_stream, 0, SEEK_SET);
                    fseek(train_labels_stream, 0, SEEK_SET);

                    // matrix_print(weight_hidden_output);
                }

            for (int i = 0; i < TRAINING_SET_SIZE; i++)
            {

                // Feed forward process

                // initialisation of input layer

                input_array = csv_to_array_vectors(train_vectors_stream);
                matrix_initialize(input_layer_transpose, 1, INPUT_NEURON, input_array);
                matrix_transpose(input_layer_transpose, input_layer);
                // feed forward on hidden layer
                matrix_multiply(weight_input_hidden, input_layer, hidden_layer);
                matrix_add(hidden_layer, bias_hidden);
                reLU(hidden_layer, activation_hidden_matrix);

                // feed forward on output layer
                matrix_multiply(weight_hidden_output, activation_hidden_matrix, output_layer);
                matrix_add(output_layer, bias_output);
                reLU(output_layer, activation_output_matrix);

                // free memory
                free(input_array[0]);
                free(input_array);
            }*/
            matrix_print(output_layer);
        }

        // free block
        fclose(train_vectors_stream);
        fclose(train_labels_stream);
        matrix_free(input_layer);
        matrix_free(input_layer_transpose);
        matrix_free(hidden_layer_1);
        matrix_free(hidden_layer_2);
        matrix_free(hidden_layer_3);
        matrix_free(hidden_layer_4);
        matrix_free(activation_hidden_1_matrix);
        matrix_free(activation_hidden_2_matrix);
        matrix_free(activation_hidden_3_matrix);
        matrix_free(activation_hidden_4_matrix);
        matrix_free(output_layer);
        matrix_free(activation_output_matrix);
        /*
        matrix_free(derivate_error_output_layer);
        matrix_free(derivate_error_output_layer_diag);
        matrix_free(derivate_output);
        matrix_free(derivate_output_diag);
        matrix_free(derivate_output_activiation);
        matrix_free(derivate_error_hidden_layer);
        matrix_free(derivate_error_hidden_layer_transpose);
        matrix_free(derivate_hidden);
        matrix_free(derivate_hidden_diag);
        matrix_free(derivate_hidden_error);
        matrix_free(derivate_error_hidden_layer_diag);
        matrix_free(error_weight_gradient_output);
        matrix_free(error_weight_gradient_hidden);
        matrix_free(error_weight_gradient_bias_output);
        matrix_free(error_weight_gradient_bias_hidden);
        matrix_free(error_weight_gradient_output_step);
        matrix_free(error_weight_gradient_hidden_step);
        matrix_free(error_weight_gradient_bias_output_step);
        matrix_free(error_weight_gradient_bias_hidden_step);
        matrix_free(activation_input_matrix);
        matrix_free(error_weight_gradient_output_step_transpose);
        matrix_free(activation_input_matrix_transpose);*/
        matrix_free(bias_hidden_1);
        matrix_free(bias_hidden_2);
        matrix_free(bias_hidden_3);
        matrix_free(bias_hidden_4);
        matrix_free(bias_output);
        matrix_free(weight_input_hidden_1);
        matrix_free(weight_input_hidden_2);
        matrix_free(weight_input_hidden_3);
        matrix_free(weight_input_hidden_4);
        matrix_free(weight_hidden_4_output);
        printf("end\n");
        return 0;
    }
}