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
#define EPOCH 5

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
    float **expected_output_array;
    float learning_rate = 0.5;
    int test = 0;
    int cpt = 0;
    // feed forward matrix
    matrix_t *input_layer_transpose = matrix_create(1, INPUT_NEURON);
    matrix_t *input_layer = matrix_create(INPUT_NEURON, 1);
    matrix_t *hidden_layer_1 = matrix_create(HIDDEN_NEURON_1, 1);
    matrix_t *hidden_layer_2 = matrix_create(HIDDEN_NEURON_2, 1);
    matrix_t *hidden_layer_3 = matrix_create(HIDDEN_NEURON_3, 1);
    matrix_t *hidden_layer_4 = matrix_create(HIDDEN_NEURON_4, 1);
    matrix_t *output_layer = matrix_create(OUTPUT_NEURON, 1);

    matrix_t *bias_hidden_1 = matrix_create(HIDDEN_NEURON_1, 1);
    matrix_initialize_random(bias_hidden_1);
    matrix_t *bias_hidden_2 = matrix_create(HIDDEN_NEURON_2, 1);
    matrix_initialize_random(bias_hidden_2);
    matrix_t *bias_hidden_3 = matrix_create(HIDDEN_NEURON_3, 1);
    matrix_initialize_random(bias_hidden_3);
    matrix_t *bias_hidden_4 = matrix_create(HIDDEN_NEURON_4, 1);
    matrix_initialize_random(bias_hidden_4);

    matrix_t *bias_output = matrix_create(OUTPUT_NEURON, 1);
    matrix_initialize_random(bias_output);

    matrix_t *weight_input_hidden_1 = matrix_create(HIDDEN_NEURON_1, INPUT_NEURON);
    matrix_t *weight_input_hidden_1_transpose = matrix_create(INPUT_NEURON, HIDDEN_NEURON_1);
    matrix_initialize_random(weight_input_hidden_1);
    matrix_t *weight_input_hidden_2 = matrix_create(HIDDEN_NEURON_2, HIDDEN_NEURON_1);
    matrix_t *weight_input_hidden_2_transpose = matrix_create(HIDDEN_NEURON_1, HIDDEN_NEURON_2);
    matrix_initialize_random(weight_input_hidden_2);
    matrix_t *weight_input_hidden_3 = matrix_create(HIDDEN_NEURON_3, HIDDEN_NEURON_2);
    matrix_t *weight_input_hidden_3_transpose = matrix_create(HIDDEN_NEURON_2, HIDDEN_NEURON_3);
    matrix_initialize_random(weight_input_hidden_3);
    matrix_t *weight_input_hidden_4 = matrix_create(HIDDEN_NEURON_4, HIDDEN_NEURON_3);
    matrix_t *weight_input_hidden_4_transpose = matrix_create(HIDDEN_NEURON_3, HIDDEN_NEURON_4);
    matrix_initialize_random(weight_input_hidden_4);
    matrix_t *weight_hidden_4_output = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON_4);
    matrix_t *weight_hidden_4_output_transpose = matrix_create(HIDDEN_NEURON_4, OUTPUT_NEURON);
    matrix_initialize_random(weight_hidden_4_output);

    matrix_t *activation_hidden_1_matrix = matrix_create(HIDDEN_NEURON_1, 1);
    matrix_t *activation_hidden_1_matrix_transpose = matrix_create(1, HIDDEN_NEURON_1);

    matrix_t *activation_hidden_2_matrix = matrix_create(HIDDEN_NEURON_2, 1);
    matrix_t *activation_hidden_2_matrix_transpose = matrix_create(1, HIDDEN_NEURON_2);

    matrix_t *activation_hidden_3_matrix = matrix_create(HIDDEN_NEURON_3, 1);
    matrix_t *activation_hidden_3_matrix_transpose = matrix_create(1, HIDDEN_NEURON_3);

    matrix_t *activation_hidden_4_matrix = matrix_create(HIDDEN_NEURON_4, 1);

    matrix_t *activation_output_matrix = matrix_create(OUTPUT_NEURON, 1);

    // backpropagation matrix

    matrix_t *expected_output = matrix_create(1, OUTPUT_NEURON);
    matrix_t *derivate_error_output_layer = matrix_create(OUTPUT_NEURON, 1);
    matrix_t *derivate_output = matrix_create(OUTPUT_NEURON, 1);
    matrix_t *derivate_output_diag = matrix_create(OUTPUT_NEURON, OUTPUT_NEURON);
    matrix_t *derivate_error_output_layer_diag = matrix_create(OUTPUT_NEURON, OUTPUT_NEURON);
    matrix_t *derivate_output_activiation = matrix_create(OUTPUT_NEURON, 1);
    matrix_t *derivate_output_activiation_transpose = matrix_create(1, OUTPUT_NEURON);

    matrix_t *derivate_error_hidden_4_layer = matrix_create(HIDDEN_NEURON_4, 1);
    matrix_t *derivate_error_hidden_4_layer_transpose = matrix_create(1, HIDDEN_NEURON_4);
    matrix_t *derivate_error_hidden_4_layer_diag = matrix_create(HIDDEN_NEURON_4, HIDDEN_NEURON_4);

    matrix_t *derivate_error_hidden_3_layer = matrix_create(HIDDEN_NEURON_3, 1);
    matrix_t *derivate_error_hidden_3_layer_transpose = matrix_create(1, HIDDEN_NEURON_3);
    matrix_t *derivate_error_hidden_3_layer_diag = matrix_create(HIDDEN_NEURON_3, HIDDEN_NEURON_3);

    matrix_t *derivate_error_hidden_2_layer = matrix_create(HIDDEN_NEURON_2, 1);
    matrix_t *derivate_error_hidden_2_layer_transpose = matrix_create(1, HIDDEN_NEURON_2);
    matrix_t *derivate_error_hidden_2_layer_diag = matrix_create(HIDDEN_NEURON_2, HIDDEN_NEURON_2);

    matrix_t *derivate_error_hidden_1_layer = matrix_create(HIDDEN_NEURON_1, 1);
    matrix_t *derivate_error_hidden_1_layer_transpose = matrix_create(1, HIDDEN_NEURON_1);
    matrix_t *derivate_error_hidden_1_layer_diag = matrix_create(HIDDEN_NEURON_1, HIDDEN_NEURON_1);

    matrix_t *error_weight_gradient_hidden_4 = matrix_create(HIDDEN_NEURON_4, HIDDEN_NEURON_3);
    matrix_t *error_weight_gradient_hidden_3 = matrix_create(HIDDEN_NEURON_3, HIDDEN_NEURON_2);
    matrix_t *error_weight_gradient_hidden_2 = matrix_create(HIDDEN_NEURON_2, HIDDEN_NEURON_1);
    matrix_t *error_weight_gradient_hidden_1 = matrix_create(HIDDEN_NEURON_1, INPUT_NEURON);

    matrix_t *error_weight_gradient_output = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON_4);
    matrix_t *error_weight_gradient_output_step_transpose = matrix_create(HIDDEN_NEURON_4, OUTPUT_NEURON);
    matrix_t *error_weight_gradient_output_step = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON_4);

    matrix_t *error_weight_gradient_bias_output_step = matrix_create(OUTPUT_NEURON, 1);
    matrix_t *error_weight_gradient_bias_output = matrix_create(OUTPUT_NEURON, 1);
    matrix_t *error_weight_gradient_bias_hidden_4 = matrix_create(HIDDEN_NEURON_4, 1);
    matrix_t *error_weight_gradient_bias_hidden_3 = matrix_create(HIDDEN_NEURON_3, 1);
    matrix_t *error_weight_gradient_bias_hidden_2 = matrix_create(HIDDEN_NEURON_2, 1);
    matrix_t *error_weight_gradient_bias_hidden_1 = matrix_create(HIDDEN_NEURON_1, 1);

    matrix_t *derivate_hidden_4 = matrix_create(HIDDEN_NEURON_4, 1);
    matrix_t *derivate_hidden_4_activation = matrix_create(HIDDEN_NEURON_4, 1);
    matrix_t *derivate_hidden_4_diag = matrix_create(HIDDEN_NEURON_4, HIDDEN_NEURON_4);
    matrix_t *derivate_hidden_4_error = matrix_create(HIDDEN_NEURON_4, 1);

    matrix_t *derivate_hidden_3 = matrix_create(HIDDEN_NEURON_3, 1);
    matrix_t *derivate_hidden_3_activation = matrix_create(HIDDEN_NEURON_3, 1);
    matrix_t *derivate_hidden_3_diag = matrix_create(HIDDEN_NEURON_3, HIDDEN_NEURON_3);
    matrix_t *derivate_hidden_3_error = matrix_create(HIDDEN_NEURON_3, 1);

    matrix_t *derivate_hidden_2 = matrix_create(HIDDEN_NEURON_2, 1);
    matrix_t *derivate_hidden_2_activation = matrix_create(HIDDEN_NEURON_2, 1);
    matrix_t *derivate_hidden_2_diag = matrix_create(HIDDEN_NEURON_2, HIDDEN_NEURON_2);
    matrix_t *derivate_hidden_2_error = matrix_create(HIDDEN_NEURON_2, 1);

    matrix_t *derivate_hidden_1 = matrix_create(HIDDEN_NEURON_1, 1);
    matrix_t *derivate_hidden_1_diag = matrix_create(HIDDEN_NEURON_1, HIDDEN_NEURON_1);
    matrix_t *derivate_hidden_1_error = matrix_create(HIDDEN_NEURON_1, 1);

    matrix_t *activation_input_matrix = matrix_create(INPUT_NEURON, 1);
    matrix_t *activation_input_matrix_transpose = matrix_create(1, INPUT_NEURON);

    matrix_t *error_weight_gradient_hidden_4_step = matrix_create(HIDDEN_NEURON_4, HIDDEN_NEURON_3);

    matrix_t *error_weight_gradient_hidden_3_step = matrix_create(HIDDEN_NEURON_3, HIDDEN_NEURON_2);

    matrix_t *error_weight_gradient_hidden_2_step = matrix_create(HIDDEN_NEURON_2, HIDDEN_NEURON_1);

    matrix_t *error_weight_gradient_hidden_1_step = matrix_create(HIDDEN_NEURON_1, INPUT_NEURON);

    matrix_t *error_weight_gradient_bias_hidden_4_step = matrix_create(HIDDEN_NEURON_4, 1);
    matrix_t *error_weight_gradient_bias_hidden_3_step = matrix_create(HIDDEN_NEURON_3, 1);
    matrix_t *error_weight_gradient_bias_hidden_2_step = matrix_create(HIDDEN_NEURON_2, 1);
    matrix_t *error_weight_gradient_bias_hidden_1_step = matrix_create(HIDDEN_NEURON_1, 1);

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
            // error function gradiant

            // yj - dkj for j in Y (output layer)
            // matrix_initialize(expected_output_step, expected_output_step->rows, expected_output_step->cols, &expected_output->data[i][0]);
            expected_output_array = csv_to_array_labels(train_labels_stream);
            matrix_initialize(expected_output, expected_output->rows, expected_output->cols, expected_output_array);
            matrix_transpose(expected_output, derivate_error_output_layer);
            matrix_subtract(derivate_error_output_layer, activation_output_matrix);
            matrix_multiply_constant(derivate_error_output_layer, -1.0);
            // dEk/dyj for j in Y (output layer)

            // dEk/dwij for j in Y(output layer)
            matrix_transpose(derivate_output_activiation, derivate_output_activiation_transpose);
            matrix_multiply(activation_hidden_4_matrix, derivate_output_activiation_transpose, error_weight_gradient_output_step_transpose);
            matrix_transpose(error_weight_gradient_output_step_transpose, error_weight_gradient_output_step); // transpose so it is of the form of weight matrix
            matrix_add(error_weight_gradient_output, error_weight_gradient_output_step);                      // sum of each training set

            // dEk/dyj for j in Z \ (Y U X) (hidden layer 4)

            reLU_derivate(output_layer, derivate_output);
            matrix_diagonalize(derivate_output, derivate_output_diag);
            matrix_multiply(derivate_output_diag, derivate_error_output_layer, derivate_output_activiation); // dEk/dyr * sigma'r(xi r) for r in j->
            matrix_transpose(weight_hidden_4_output, weight_hidden_4_output_transpose);
            matrix_multiply(weight_hidden_4_output_transpose, derivate_output_activiation, derivate_error_hidden_4_layer); // dEk/dyj
            matrix_transpose(derivate_error_hidden_4_layer, derivate_error_hidden_4_layer_transpose);

            // dEk/dwij for j in Z \ (Y U X) (hidden layer) 4

            reLU_derivate(hidden_layer_4, derivate_hidden_4);
            matrix_diagonalize(derivate_hidden_4, derivate_hidden_4_diag);
            matrix_multiply(derivate_hidden_4_diag, derivate_error_hidden_4_layer, derivate_hidden_4_error);
            matrix_transpose(activation_hidden_3_matrix, activation_hidden_3_matrix_transpose);
            matrix_multiply(derivate_hidden_4_error, activation_hidden_3_matrix_transpose, error_weight_gradient_hidden_4_step);
            matrix_add(error_weight_gradient_hidden_4, error_weight_gradient_hidden_4_step);

            // dEk/dyj for j in Z \ (Y U X) (hidden layer 3)

            matrix_multiply(derivate_hidden_4_diag, derivate_error_hidden_4_layer, derivate_hidden_4_activation); // dEk/dyr * sigma'r(xi r) for r in j->
            matrix_transpose(weight_input_hidden_4, weight_input_hidden_4_transpose);
            matrix_multiply(weight_input_hidden_4_transpose, derivate_hidden_4_activation, derivate_error_hidden_3_layer); // dEk/dyj
            matrix_transpose(derivate_error_hidden_3_layer, derivate_error_hidden_3_layer_transpose);

            // dEk/dwij for j in Z \ (Y U X) (hidden layer) 3
            reLU_derivate(hidden_layer_3, derivate_hidden_3);
            matrix_diagonalize(derivate_hidden_3, derivate_hidden_3_diag);
            matrix_multiply(derivate_hidden_3_diag, derivate_error_hidden_3_layer, derivate_hidden_3_error);
            matrix_transpose(activation_hidden_2_matrix, activation_hidden_2_matrix_transpose);
            matrix_multiply(derivate_hidden_3_error, activation_hidden_2_matrix_transpose, error_weight_gradient_hidden_3_step);
            matrix_add(error_weight_gradient_hidden_3, error_weight_gradient_hidden_3_step);

            // dEK/dyj for j in Z \ (Y U X) (hidden layer 2)

            matrix_diagonalize(derivate_hidden_3, derivate_hidden_3_diag);
            matrix_multiply(derivate_hidden_3_diag, derivate_error_hidden_3_layer, derivate_hidden_3_activation);
            matrix_transpose(weight_input_hidden_3, weight_input_hidden_3_transpose);
            matrix_multiply(weight_input_hidden_3_transpose, derivate_hidden_3_activation, derivate_error_hidden_2_layer); // dEk/dyj
            matrix_transpose(derivate_error_hidden_2_layer, derivate_error_hidden_2_layer_transpose);

            // dEk/dwij for j in Z \ (Y U X) (hidden layer) 2
            reLU_derivate(hidden_layer_2, derivate_hidden_2);
            matrix_diagonalize(derivate_hidden_2, derivate_hidden_2_diag);
            matrix_multiply(derivate_hidden_2_diag, derivate_error_hidden_2_layer, derivate_hidden_2_error);
            matrix_transpose(activation_hidden_1_matrix, activation_hidden_1_matrix_transpose);
            matrix_multiply(derivate_hidden_2_error, activation_hidden_1_matrix_transpose, error_weight_gradient_hidden_2_step);
            matrix_add(error_weight_gradient_hidden_2, error_weight_gradient_hidden_2_step);

            // dEK/dyj for j in Z \ (Y U X) (hidden layer 1)

            matrix_diagonalize(derivate_hidden_2, derivate_hidden_2_diag);
            matrix_multiply(derivate_hidden_2_diag, derivate_error_hidden_2_layer, derivate_hidden_2_activation);
            matrix_transpose(weight_input_hidden_2, weight_input_hidden_2_transpose);
            matrix_multiply(weight_input_hidden_2_transpose, derivate_hidden_2_activation, derivate_error_hidden_1_layer); // dEk/dyj
            matrix_transpose(derivate_error_hidden_1_layer, derivate_error_hidden_1_layer_transpose);

            // dEk/dwij for j in Z \ (Y U X) (hidden layer) 1
            reLU_derivate(hidden_layer_1, derivate_hidden_1);
            matrix_diagonalize(derivate_hidden_1, derivate_hidden_1_diag);
            matrix_multiply(derivate_hidden_1_diag, derivate_error_hidden_1_layer, derivate_hidden_1_error);
            matrix_transpose(activation_input_matrix, activation_input_matrix_transpose);
            matrix_multiply(derivate_hidden_1_error, activation_input_matrix_transpose, error_weight_gradient_hidden_1_step);
            matrix_add(error_weight_gradient_hidden_1, error_weight_gradient_hidden_1_step);

            // bias of hidden layer 4 update
            matrix_diagonalize(derivate_error_hidden_4_layer, derivate_error_hidden_4_layer_diag);
            matrix_multiply(derivate_error_hidden_4_layer_diag, bias_hidden_4, error_weight_gradient_bias_hidden_4_step);
            matrix_add(error_weight_gradient_bias_hidden_4, error_weight_gradient_bias_hidden_4_step);

            // bias of hidden layer 3 update
            matrix_diagonalize(derivate_error_hidden_3_layer, derivate_error_hidden_3_layer_diag);
            matrix_multiply(derivate_error_hidden_3_layer_diag, bias_hidden_3, error_weight_gradient_bias_hidden_3_step);
            matrix_add(error_weight_gradient_bias_hidden_3, error_weight_gradient_bias_hidden_3_step);

            // bias of hidden layer 2 update
            matrix_diagonalize(derivate_error_hidden_2_layer, derivate_error_hidden_2_layer_diag);
            matrix_multiply(derivate_error_hidden_2_layer_diag, bias_hidden_2, error_weight_gradient_bias_hidden_2_step);
            matrix_add(error_weight_gradient_bias_hidden_2, error_weight_gradient_bias_hidden_2_step);

            // bias of hidden layer 1 update
            matrix_diagonalize(derivate_error_hidden_1_layer, derivate_error_hidden_1_layer_diag);
            matrix_multiply(derivate_error_hidden_1_layer_diag, bias_hidden_1, error_weight_gradient_bias_hidden_1_step);
            matrix_add(error_weight_gradient_bias_hidden_1, error_weight_gradient_bias_hidden_1_step);

            // bias of output layer update

            matrix_diagonalize(derivate_error_output_layer, derivate_error_output_layer_diag);
            matrix_multiply(derivate_error_output_layer_diag, bias_output, error_weight_gradient_bias_output_step); // dek/dwij
            matrix_add(error_weight_gradient_bias_output, error_weight_gradient_bias_output_step);

            // free memory
            free(expected_output_array[0]);
            free(expected_output_array);
        }
        printf("\n********* %d *********\n", j);
        matrix_print(output_layer);
        // update bias weight
        matrix_multiply_constant(error_weight_gradient_bias_output, -(1.0 / TRAINING_SET_SIZE) * learning_rate);
        matrix_multiply_constant(error_weight_gradient_bias_hidden_4, -(1.0 / TRAINING_SET_SIZE) * learning_rate);
        matrix_multiply_constant(error_weight_gradient_bias_hidden_3, -(1.0 / TRAINING_SET_SIZE) * learning_rate);
        matrix_multiply_constant(error_weight_gradient_bias_hidden_2, -(1.0 / TRAINING_SET_SIZE) * learning_rate);
        matrix_multiply_constant(error_weight_gradient_bias_hidden_1, -(1.0 / TRAINING_SET_SIZE) * learning_rate);

        matrix_add(bias_output, error_weight_gradient_bias_output);
        matrix_add(bias_hidden_4, error_weight_gradient_bias_hidden_4);
        matrix_add(bias_hidden_3, error_weight_gradient_bias_hidden_3);
        matrix_add(bias_hidden_2, error_weight_gradient_bias_hidden_2);
        matrix_add(bias_hidden_1, error_weight_gradient_bias_hidden_1);

        matrix_reset(error_weight_gradient_bias_output);
        matrix_reset(error_weight_gradient_bias_hidden_4);
        matrix_reset(error_weight_gradient_bias_hidden_3);
        matrix_reset(error_weight_gradient_bias_hidden_2);
        matrix_reset(error_weight_gradient_bias_hidden_1);

        // update weight

        matrix_multiply_constant(error_weight_gradient_output, -(1.0 / TRAINING_SET_SIZE) * learning_rate);
        matrix_multiply_constant(error_weight_gradient_hidden_4, -(1.0 / TRAINING_SET_SIZE) * learning_rate);
        matrix_multiply_constant(error_weight_gradient_hidden_3, -(1.0 / TRAINING_SET_SIZE) * learning_rate);
        matrix_multiply_constant(error_weight_gradient_hidden_2, -(1.0 / TRAINING_SET_SIZE) * learning_rate);
        matrix_multiply_constant(error_weight_gradient_hidden_1, -(1.0 / TRAINING_SET_SIZE) * learning_rate);

        matrix_add(weight_hidden_4_output, error_weight_gradient_output);
        matrix_add(weight_input_hidden_4, error_weight_gradient_hidden_4);
        matrix_add(weight_input_hidden_3, error_weight_gradient_hidden_3);
        matrix_add(weight_input_hidden_2, error_weight_gradient_hidden_2);
        matrix_add(weight_input_hidden_1, error_weight_gradient_hidden_1);

        matrix_reset(error_weight_gradient_output);
        matrix_reset(error_weight_gradient_hidden_4);
        matrix_reset(error_weight_gradient_hidden_3);
        matrix_reset(error_weight_gradient_hidden_2);
        matrix_reset(error_weight_gradient_hidden_1);

        // reset file pointer
        fseek(train_vectors_stream, 0, SEEK_SET);
        fseek(train_labels_stream, 0, SEEK_SET);
    }

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

        test = csv_to_array_labels_int(train_labels_stream);
        if (get_label(activation_output_matrix) == test)
            cpt++;
        free(input_array[0]);
        free(input_array);
    }
    printf("accuracy : %f\n", (float)cpt / (float)TRAINING_SET_SIZE);
    // free block

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
    matrix_free(derivate_error_output_layer);
    matrix_free(derivate_error_output_layer_diag);
    matrix_free(derivate_output);
    matrix_free(derivate_output_diag);
    matrix_free(derivate_output_activiation);
    matrix_free(error_weight_gradient_output);
    matrix_free(error_weight_gradient_bias_output);
    matrix_free(error_weight_gradient_output_step);
    matrix_free(error_weight_gradient_bias_output_step);
    matrix_free(activation_input_matrix);
    matrix_free(error_weight_gradient_output_step_transpose);
    matrix_free(activation_input_matrix_transpose);
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
    matrix_free(error_weight_gradient_hidden_1);
    matrix_free(error_weight_gradient_bias_hidden_1);
    matrix_free(error_weight_gradient_hidden_1_step);
    matrix_free(error_weight_gradient_bias_hidden_1_step);
    matrix_free(error_weight_gradient_hidden_2);
    matrix_free(error_weight_gradient_bias_hidden_2);
    matrix_free(error_weight_gradient_hidden_2_step);
    matrix_free(error_weight_gradient_bias_hidden_2_step);
    matrix_free(error_weight_gradient_hidden_3);
    matrix_free(error_weight_gradient_bias_hidden_3);
    matrix_free(error_weight_gradient_hidden_3_step);
    matrix_free(error_weight_gradient_bias_hidden_3_step);
    matrix_free(error_weight_gradient_hidden_4);
    matrix_free(error_weight_gradient_bias_hidden_4);
    matrix_free(error_weight_gradient_hidden_4_step);
    matrix_free(error_weight_gradient_bias_hidden_4_step);
    matrix_free(weight_input_hidden_1_transpose);
    matrix_free(weight_input_hidden_2_transpose);
    matrix_free(weight_input_hidden_3_transpose);
    matrix_free(weight_input_hidden_4_transpose);
    matrix_free(weight_hidden_4_output_transpose);
    matrix_free(activation_hidden_1_matrix_transpose);
    matrix_free(activation_hidden_2_matrix_transpose);
    matrix_free(activation_hidden_3_matrix_transpose);
    matrix_free(expected_output);
    matrix_free(derivate_output_activiation_transpose);
    matrix_free(derivate_error_hidden_4_layer);
    matrix_free(derivate_error_hidden_3_layer);
    matrix_free(derivate_error_hidden_2_layer);
    matrix_free(derivate_error_hidden_1_layer);
    matrix_free(derivate_error_hidden_4_layer_transpose);
    matrix_free(derivate_error_hidden_3_layer_transpose);
    matrix_free(derivate_error_hidden_2_layer_transpose);
    matrix_free(derivate_error_hidden_1_layer_transpose);
    matrix_free(derivate_error_hidden_1_layer_diag);
    matrix_free(derivate_error_hidden_2_layer_diag);
    matrix_free(derivate_error_hidden_3_layer_diag);
    matrix_free(derivate_error_hidden_4_layer_diag);
    matrix_free(derivate_hidden_4);
    matrix_free(derivate_hidden_3);
    matrix_free(derivate_hidden_2);
    matrix_free(derivate_hidden_1);
    matrix_free(derivate_hidden_4_diag);
    matrix_free(derivate_hidden_3_diag);
    matrix_free(derivate_hidden_2_diag);
    matrix_free(derivate_hidden_1_diag);
    matrix_free(derivate_hidden_4_activation);
    matrix_free(derivate_hidden_3_activation);
    matrix_free(derivate_hidden_2_activation);
    matrix_free(derivate_hidden_1_error);
    matrix_free(derivate_hidden_3_error);
    matrix_free(derivate_hidden_2_error);
    matrix_free(derivate_hidden_4_error);

    printf("end\n");
    return 0;
}
