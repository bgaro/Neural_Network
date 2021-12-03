#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "matrix.h"
#include "activation.h"
#include "csv_to_array.h"
#include "neural_network.h"

#define INPUT_NEURON 784
#define HIDDEN_NEURON 110
#define HIDDEN_NEURON_1 90
#define OUTPUT_NEURON 10
#define LAYER_NUM 4
#define TRAINING_SET_SIZE 1000
#define TEST_SET_SIZE 10000
#define OUPUT_SIZE 1
#define EPOCH 100
int main()
{
    srand(time(NULL));
    FILE *train_vectors_stream = fopen("./data/fashion_mnist_train_vectors.csv", "r");
    if (train_vectors_stream == NULL)
    {
        printf("Error opening file\n");
        return 1;
    }
    FILE *train_labels_stream = fopen("./data/fashion_mnist_train_labels.csv", "r");
    if (train_labels_stream == NULL)
    {
        printf("Error opening file\n");
        return 1;
    }

    float **input_array;
    float **expected_output_array;
    float learning_rate = 0.14;
    float alpha = 0.9;
    float error = 0.0;
    int test = 0;
    int cpt = 0;
    // feed forward matrix
    matrix_t *input_layer_transpose = matrix_create(1, INPUT_NEURON);
    matrix_t *input_layer = matrix_create(INPUT_NEURON, 1);
    matrix_t *hidden_layer = matrix_create(HIDDEN_NEURON, 1);
    matrix_t *hidden_layer_1 = matrix_create(HIDDEN_NEURON_1, 1);
    matrix_t *output_layer = matrix_create(OUTPUT_NEURON, 1);

    matrix_t *bias_hidden = matrix_create(HIDDEN_NEURON, 1);
    matrix_t *bias_hidden_1 = matrix_create(HIDDEN_NEURON_1, 1);

    matrix_t *bias_output = matrix_create(OUTPUT_NEURON, 1);

    matrix_t *weight_input_hidden = matrix_create(HIDDEN_NEURON_1, INPUT_NEURON);
    matrix_t *weight_input_hidden_transpose = matrix_create(INPUT_NEURON, HIDDEN_NEURON_1);
    matrix_initialize_random(weight_input_hidden, HIDDEN_NEURON_1, INPUT_NEURON);

    matrix_t *weight_hidden_hidden = matrix_create(HIDDEN_NEURON, HIDDEN_NEURON_1);
    matrix_initialize_random(weight_hidden_hidden, HIDDEN_NEURON, HIDDEN_NEURON_1);

    matrix_t *weight_hidden_output = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON);
    matrix_t *weight_hidden_output_transpose = matrix_create(HIDDEN_NEURON, OUTPUT_NEURON);
    matrix_initialize_random(weight_hidden_output, OUTPUT_NEURON, HIDDEN_NEURON);

    matrix_t *activation_hidden_matrix = matrix_create(HIDDEN_NEURON, 1);
    matrix_t *activation_hidden_matrix_transpose = matrix_create(1, HIDDEN_NEURON);

    matrix_t *activation_hidden_1_matrix = matrix_create(HIDDEN_NEURON_1, 1);
    matrix_t *activation_hidden_1_matrix_transpose = matrix_create(1, HIDDEN_NEURON_1);

    matrix_t *activation_output_matrix = matrix_create(OUTPUT_NEURON, 1);

    // backpropagation matrix

    matrix_t *expected_output = matrix_create(1, OUTPUT_NEURON);
    matrix_t *derivate_error_output_layer = matrix_create(OUTPUT_NEURON, 1);
    matrix_t *derivate_error_activation_output = matrix_create(OUTPUT_NEURON, 1);

    matrix_t *derivate_output = matrix_create(OUTPUT_NEURON, OUTPUT_NEURON);
    matrix_t *derivate_output_diag = matrix_create(OUTPUT_NEURON, OUTPUT_NEURON);
    matrix_t *derivate_error_output_layer_diag = matrix_create(OUTPUT_NEURON, OUTPUT_NEURON);
    matrix_t *derivate_output_activiation = matrix_create(OUTPUT_NEURON, 1);
    matrix_t *derivate_output_activiation_transpose = matrix_create(1, OUTPUT_NEURON);

    matrix_t *derivate_error_hidden_layer = matrix_create(HIDDEN_NEURON, 1);
    matrix_t *derivate_error_hidden_layer_transpose = matrix_create(1, HIDDEN_NEURON);
    matrix_t *derivate_error_hidden_layer_diag = matrix_create(HIDDEN_NEURON, HIDDEN_NEURON);

    matrix_t *derivate_error_hidden_layer_1 = matrix_create(HIDDEN_NEURON_1, 1);
    matrix_t *derivate_error_hidden_layer_1_transpose = matrix_create(1, HIDDEN_NEURON_1);

    matrix_t *error_weight_gradient_hidden = matrix_create(HIDDEN_NEURON, HIDDEN_NEURON_1);
    matrix_t *error_weight_gradient_hidden_previous_step = matrix_create(HIDDEN_NEURON, HIDDEN_NEURON_1);

    matrix_t *error_weight_gradient_hidden_1 = matrix_create(HIDDEN_NEURON_1, INPUT_NEURON);
    matrix_t *error_weight_gradient_hidden_1_previous_step = matrix_create(HIDDEN_NEURON_1, INPUT_NEURON);

    matrix_t *error_weight_gradient_output = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON);
    matrix_t *error_weight_gradient_output_step_transpose = matrix_create(HIDDEN_NEURON, OUTPUT_NEURON);
    matrix_t *error_weight_gradient_output_step = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON);
    matrix_t *error_weight_gradient_output_previous_step = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON);

    matrix_t *error_weight_gradient_bias_output_step = matrix_create(OUTPUT_NEURON, 1);
    matrix_t *error_weight_gradient_bias_output = matrix_create(OUTPUT_NEURON, 1);
    matrix_t *error_weight_gradient_bias_output_previous_step = matrix_create(OUTPUT_NEURON, 1);

    matrix_t *error_weight_gradient_bias_hidden = matrix_create(HIDDEN_NEURON, 1);
    matrix_t *error_weight_gradient_bias_hidden_previous_step = matrix_create(HIDDEN_NEURON, 1);

    matrix_t *error_weight_gradient_bias_hidden_1 = matrix_create(HIDDEN_NEURON_1, 1);
    matrix_t *error_weight_gradient_bias_hidden_1_previous_step = matrix_create(HIDDEN_NEURON_1, 1);

    matrix_t *derivate_hidden = matrix_create(HIDDEN_NEURON, 1);
    matrix_t *derivate_hidden_activation = matrix_create(HIDDEN_NEURON, 1);
    matrix_t *derivate_hidden_activation_transpose = matrix_create(1, HIDDEN_NEURON);

    matrix_t *derivate_hidden_1_activation = matrix_create(HIDDEN_NEURON_1, 1);

    matrix_t *derivate_hidden_diag = matrix_create(HIDDEN_NEURON, HIDDEN_NEURON);
    matrix_t *derivate_hidden_error = matrix_create(HIDDEN_NEURON, 1);

    matrix_t *derivate_hidden_1_error = matrix_create(HIDDEN_NEURON_1, 1);

    matrix_t *activation_input_matrix = matrix_create(INPUT_NEURON, 1);
    matrix_t *activation_input_matrix_transpose = matrix_create(1, INPUT_NEURON);

    matrix_t *error_weight_gradient_hidden_step = matrix_create(HIDDEN_NEURON, HIDDEN_NEURON_1);
    matrix_t *error_weight_gradient_hidden_1_step = matrix_create(HIDDEN_NEURON_1, INPUT_NEURON);

    matrix_t *error_weight_gradient_bias_hidden_step = matrix_create(HIDDEN_NEURON, 1);

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
            feed_forward(weight_input_hidden, input_layer, bias_hidden_1, hidden_layer_1, activation_hidden_1_matrix, reLU);

            // feed forward on hidden layer
            feed_forward(weight_hidden_hidden, activation_hidden_1_matrix, bias_hidden, hidden_layer, activation_hidden_matrix, reLU);

            // feed forward on output layer
            feed_forward(weight_hidden_output, activation_hidden_matrix, bias_output, output_layer, activation_output_matrix, softmax);

            free(input_array[0]);
            free(input_array);
            // error function gradiant

            // yj - dkj for j in Y (output layer)
            // matrix_initialize(expected_output_step, expected_output_step->rows, expected_output_step->cols, &expected_output->data[i][0]);
            expected_output_array = csv_to_array_labels(train_labels_stream);
            matrix_initialize(expected_output, expected_output->rows, expected_output->cols, expected_output_array);
            matrix_transpose(expected_output, derivate_error_output_layer);
            matrix_subtract(derivate_error_output_layer, activation_output_matrix);
            matrix_multiply_constant(derivate_error_output_layer, -(1.0 / TRAINING_SET_SIZE));
            softmax_derivate(activation_output_matrix, derivate_output);

            for (int d = 0; d < OUTPUT_NEURON; d++)
            {
                error += expected_output_array[0][d] * log(activation_output_matrix->data[d]);
            }

            // dEk/dyj for j in Y (output layer)

            // dEk/dyj for j in Z \ (Y U X) (hidden layer)
            backward_propagation_neurons(derivate_error_output_layer, derivate_output, derivate_output_activiation, derivate_output_activiation_transpose, weight_hidden_output, derivate_error_hidden_layer_transpose, derivate_error_hidden_layer, SOFTMAX);

            // dEk/dwij for j in Z \ (Y U X) (hidden layer) 3
            reLU_derivate(hidden_layer, derivate_hidden);
            backward_propagation_neurons(derivate_error_hidden_layer, derivate_hidden, derivate_hidden_activation, derivate_hidden_activation_transpose, weight_hidden_hidden, derivate_error_hidden_layer_1_transpose, derivate_error_hidden_layer_1, RELU);

            // dEk/dwij for j in Y(output layer)

            backward_propagation_weights(derivate_error_output_layer, derivate_output, derivate_error_activation_output, activation_hidden_matrix, activation_hidden_matrix_transpose, error_weight_gradient_output_step, SOFTMAX);
            matrix_add(error_weight_gradient_output, error_weight_gradient_output_step); // sum of each training set

            // dEk/dwij for j in Z \ (Y U X) (hidden layer) 4

            backward_propagation_weights(derivate_error_hidden_layer, derivate_hidden, derivate_hidden_error, activation_hidden_1_matrix, activation_hidden_1_matrix_transpose, error_weight_gradient_hidden_step, RELU);
            matrix_add(error_weight_gradient_hidden, error_weight_gradient_hidden_step);

            // dEk/dwij for j in Z \ (Y U X) (hidden layer) 3

            reLU_derivate(hidden_layer_1, derivate_hidden_1_activation);
            backward_propagation_weights(derivate_error_hidden_layer_1, derivate_hidden_1_activation, derivate_hidden_1_error, input_layer, input_layer_transpose, error_weight_gradient_hidden_1_step, RELU);
            matrix_add(error_weight_gradient_hidden_1, error_weight_gradient_hidden_1_step);

            // bias of hidden layer 3 update
            matrix_add(error_weight_gradient_bias_hidden_1, derivate_hidden_1_error);

            // bias of hidden layer 4 update
            matrix_add(error_weight_gradient_bias_hidden, derivate_hidden_error);
            // bias of output layer update

            matrix_add(error_weight_gradient_bias_output, derivate_error_activation_output);
            // free memory
            free(expected_output_array[0]);
            free(expected_output_array);
        }
        printf("*************** EPOCH %d *************\n", j);
        printf("Error: %f\n", -error);
        error = 0.0;
        // update bias weight
        matrix_multiply_constant(error_weight_gradient_bias_output, -learning_rate);
        matrix_multiply_constant(error_weight_gradient_bias_hidden, -learning_rate);
        matrix_multiply_constant(error_weight_gradient_bias_hidden_1, -learning_rate);

        matrix_multiply_constant(error_weight_gradient_bias_output_previous_step, alpha);
        matrix_multiply_constant(error_weight_gradient_bias_hidden_previous_step, alpha);
        matrix_multiply_constant(error_weight_gradient_bias_hidden_1_previous_step, alpha);

        matrix_add(error_weight_gradient_bias_output, error_weight_gradient_bias_output_previous_step);
        matrix_add(error_weight_gradient_bias_hidden, error_weight_gradient_bias_hidden_previous_step);
        matrix_add(error_weight_gradient_bias_hidden_1, error_weight_gradient_bias_hidden_1_previous_step);

        matrix_copy(error_weight_gradient_bias_output, error_weight_gradient_bias_output_previous_step);
        matrix_copy(error_weight_gradient_bias_hidden, error_weight_gradient_bias_hidden_previous_step);
        matrix_copy(error_weight_gradient_bias_hidden_1, error_weight_gradient_bias_hidden_1_previous_step);

        matrix_add(bias_output, error_weight_gradient_bias_output);
        matrix_add(bias_hidden, error_weight_gradient_bias_hidden);
        matrix_add(bias_hidden_1, error_weight_gradient_bias_hidden_1);

        matrix_reset(error_weight_gradient_bias_output);
        matrix_reset(error_weight_gradient_bias_hidden);
        matrix_reset(error_weight_gradient_bias_hidden_1);

        // update weight

        matrix_multiply_constant(error_weight_gradient_output, -learning_rate);
        matrix_multiply_constant(error_weight_gradient_hidden, -learning_rate);
        matrix_multiply_constant(error_weight_gradient_hidden_1, -learning_rate);

        matrix_multiply_constant(error_weight_gradient_output_previous_step, alpha);
        matrix_multiply_constant(error_weight_gradient_hidden_previous_step, alpha);
        matrix_multiply_constant(error_weight_gradient_hidden_1_previous_step, alpha);

        matrix_add(error_weight_gradient_output, error_weight_gradient_output_previous_step);
        matrix_add(error_weight_gradient_hidden, error_weight_gradient_hidden_previous_step);
        matrix_add(error_weight_gradient_hidden_1, error_weight_gradient_hidden_1_previous_step);

        matrix_copy(error_weight_gradient_output, error_weight_gradient_output_previous_step);
        matrix_copy(error_weight_gradient_hidden, error_weight_gradient_hidden_previous_step);
        matrix_copy(error_weight_gradient_hidden_1, error_weight_gradient_hidden_1_previous_step);

        matrix_add(weight_hidden_output, error_weight_gradient_output);
        matrix_add(weight_input_hidden, error_weight_gradient_hidden_1);
        matrix_add(weight_hidden_hidden, error_weight_gradient_hidden);

        matrix_reset(error_weight_gradient_output);
        matrix_reset(error_weight_gradient_hidden);
        matrix_reset(error_weight_gradient_hidden_1);
        printf("Learning rate: %f\n", learning_rate);

        // reset file pointer
        fseek(train_vectors_stream, 0, SEEK_SET);
        fseek(train_labels_stream, 0, SEEK_SET);
    }
    fclose(train_vectors_stream);
    fclose(train_labels_stream);

    train_vectors_stream = fopen("./data/fashion_mnist_test_vectors.csv", "r");

    train_labels_stream = fopen("./data/fashion_mnist_test_labels.csv", "r");
    for (int i = 0; i < TEST_SET_SIZE; i++)
    {

        // Feed forward process

        // initialisation of input layer

        input_array = csv_to_array_vectors(train_vectors_stream);
        matrix_initialize(input_layer_transpose, 1, INPUT_NEURON, input_array);
        matrix_transpose(input_layer_transpose, input_layer);

        // feed forward on hidden layer 1
        feed_forward(weight_input_hidden, input_layer, bias_hidden_1, hidden_layer_1, activation_hidden_1_matrix, reLU);

        // feed forward on hidden layer
        feed_forward(weight_hidden_hidden, activation_hidden_1_matrix, bias_hidden, hidden_layer, activation_hidden_matrix, reLU);

        // feed forward on output layer
        feed_forward(weight_hidden_output, activation_hidden_matrix, bias_output, output_layer, activation_output_matrix, softmax);

        free(input_array[0]);
        free(input_array);
        // error function gradiant

        test = csv_to_array_labels_int(train_labels_stream);
        if (get_label(activation_output_matrix) == test)
            cpt++;
    }
    printf("%d / %d\n", cpt, TEST_SET_SIZE);
    fclose(train_vectors_stream);
    fclose(train_labels_stream);
    printf("accuracy : %f percent\n", (float)cpt / (float)TEST_SET_SIZE * 100.0);
    // free block

    matrix_free(input_layer);
    matrix_free(input_layer_transpose);
    matrix_free(hidden_layer);
    matrix_free(activation_hidden_matrix);
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
    matrix_free(bias_hidden);
    matrix_free(bias_output);
    matrix_free(weight_input_hidden);
    matrix_free(weight_hidden_output);
    matrix_free(error_weight_gradient_hidden);
    matrix_free(error_weight_gradient_bias_hidden);
    matrix_free(error_weight_gradient_hidden_step);
    matrix_free(error_weight_gradient_bias_hidden_step);
    matrix_free(weight_input_hidden_transpose);
    matrix_free(weight_hidden_output_transpose);
    matrix_free(activation_hidden_matrix_transpose);
    matrix_free(expected_output);
    matrix_free(derivate_output_activiation_transpose);
    matrix_free(derivate_error_hidden_layer);
    matrix_free(derivate_error_hidden_layer_transpose);
    matrix_free(derivate_error_hidden_layer_diag);
    matrix_free(derivate_hidden);
    matrix_free(derivate_hidden_diag);
    matrix_free(derivate_hidden_activation);
    matrix_free(derivate_hidden_error);
    matrix_free(derivate_error_activation_output);
    matrix_free(error_weight_gradient_bias_output_previous_step);
    matrix_free(error_weight_gradient_bias_hidden_previous_step);
    matrix_free(error_weight_gradient_output_previous_step);
    matrix_free(error_weight_gradient_hidden_previous_step);
    printf("end\n");
    return 0;
}
