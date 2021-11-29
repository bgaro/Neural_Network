#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "activation.h"

#define INPUT_NEURON 2
#define HIDDEN_NEURON 4
#define OUTPUT_NEURON 1
#define LAYER_NUM 3
#define TRAINING_SET_SIZE 4
#define OUPUT_SIZE 1
#define EPOCH 10000

int main()
{

    matrix_t *training_set = matrix_create(TRAINING_SET_SIZE, INPUT_NEURON);
    float training_set_data[TRAINING_SET_SIZE][INPUT_NEURON] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}};
    matrix_initialize(training_set, TRAINING_SET_SIZE, INPUT_NEURON, training_set_data);

    matrix_t *expected_output = matrix_create(TRAINING_SET_SIZE, OUTPUT_NEURON);
    float expected_output_data[TRAINING_SET_SIZE][OUTPUT_NEURON] = {
        {0},
        {1},
        {1},
        {0}};
    matrix_initialize(expected_output, TRAINING_SET_SIZE, OUTPUT_NEURON, expected_output_data);

    //feed forward matrix
    matrix_t *activation_output_matrix;
    matrix_t *activation_hidden_matrix;
    matrix_t *activation_input_matrix;
    matrix_t *activation_input_matrix_transpose;
    matrix_t *input_layer = matrix_create(INPUT_NEURON, 1);
    matrix_t *hidden_layer;
    matrix_t *output_layer;

    matrix_t *derivate_output;
    matrix_t *derivate_output_diag;
    matrix_t *derivate_hidden;
    matrix_t *derivate_hidden_error;
    matrix_t *derivate_hidden_diag;
    matrix_t *expected_output_step = matrix_create(OUTPUT_NEURON, 1);
    matrix_t *derivate_error_hidden_layer;
    matrix_t *derivate_error_hidden_layer_diag;
    matrix_t *derivate_error_hidden_layer_transpose;
    matrix_t *derivate_error_output_layer;
    matrix_t *derivate_predicted_output;
    matrix_t *derivate_error_output_layer_diag;
    matrix_t *derivate_hidden_activation;
    matrix_t *derivate_output_activiation;
    matrix_t *diagonal_error_gradient;
    matrix_t *error_weight_gradient_output = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON);
    float learning_rate = -0.1;
    matrix_t *error_weight_gradient_hidden = matrix_create(HIDDEN_NEURON, INPUT_NEURON);
    matrix_t *error_weight_gradient_output_step;
    matrix_t *error_weight_gradient_output_step_transpose;
    matrix_t *error_weight_gradient_hidden_step;
    matrix_t *error_weight_gradient_bias_output_step;
    matrix_t *error_weight_gradient_bias_hidden_step;
    matrix_t *error_weight_gradient_bias_output = matrix_create(OUTPUT_NEURON, 1);
    matrix_t *error_weight_gradient_bias_hidden = matrix_create(HIDDEN_NEURON, 1);

    matrix_t *tmp;
    matrix_t *tmp2;
    matrix_t *tmp3;
    matrix_t *tmp4;

    matrix_t *weight_input_hidden = matrix_create(HIDDEN_NEURON, INPUT_NEURON);
    matrix_initialize_random(weight_input_hidden, 100);
    matrix_t *weight_hidden_output = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON);
    matrix_initialize_random(weight_hidden_output, 100);

    matrix_t *bias_hidden = matrix_create(HIDDEN_NEURON, 1);
    matrix_initialize_random(bias_hidden, 100);
    matrix_t *bias_output = matrix_create(OUTPUT_NEURON, 1);
    matrix_initialize_random(bias_output, 100);

    // XOR initialization

    for (int j = 0; j < EPOCH; j++)
    {
        for (int i = 0; i < TRAINING_SET_SIZE; i++)
        {
            //Feed forward process

            //initialisation of input layer
            input_layer->data[0][0] = training_set->data[i][0];
            input_layer->data[1][0] = training_set->data[i][1];

            //feed forward on hidden layer
            hidden_layer = matrix_multiply(weight_input_hidden, input_layer);
            matrix_add(hidden_layer, bias_hidden);
            activation_hidden_matrix = reLU(hidden_layer);

            //feed forward on output layer
            output_layer = matrix_multiply(weight_hidden_output, activation_hidden_matrix);
            matrix_add(output_layer, bias_output);
            activation_output_matrix = reLU(output_layer);
            // error function gradiant

            // yj - dkj for j in Y (output layer)
            //matrix_initialize(expected_output_step, expected_output_step->rows, expected_output_step->cols, &expected_output->data[i][0]);
            expected_output_step->data[0][0] = expected_output->data[i][0] - activation_output_matrix->data[0][0];

            // dEk/dyj for j in Y (output layer)
            matrix_multiply_constant(expected_output_step, -1);
            derivate_error_output_layer = matrix_copy(expected_output_step);

            // dEk/dyj for j in Z \ (Y U X) (hidden layer)

            derivate_output = reLU_derivate(output_layer);

            derivate_output_diag = matrix_diagonalize(derivate_output);
            derivate_output_activiation = matrix_multiply(derivate_output_diag, derivate_error_output_layer); // dEk/dyr * sigma'r(xi r) for r in j->
            derivate_error_hidden_layer = matrix_multiply(derivate_output_activiation, weight_hidden_output); //dEk/dyj
            derivate_error_hidden_layer_transpose = matrix_transpose(derivate_error_hidden_layer);

            // dEk/dwij for j in Y(output layer)
            error_weight_gradient_output_step_transpose = matrix_multiply(activation_hidden_matrix, derivate_output_activiation);
            error_weight_gradient_output_step = matrix_transpose(error_weight_gradient_output_step_transpose); //transpose so it is of the form of weight matrix
            matrix_add(error_weight_gradient_output, error_weight_gradient_output_step);                       //sum of each training set

            //bias of output layer update

            derivate_error_output_layer_diag = matrix_diagonalize(derivate_error_output_layer);
            error_weight_gradient_bias_output_step = matrix_multiply(derivate_error_output_layer_diag, bias_output); //dek/dwij
            matrix_add(error_weight_gradient_bias_output, error_weight_gradient_bias_output_step);

            //dEk/dwij for j in Z \ (Y U X) (hidden layer)
            derivate_hidden = reLU_derivate(hidden_layer);
            derivate_hidden_diag = matrix_diagonalize(derivate_hidden);
            derivate_hidden_error = matrix_multiply(derivate_hidden_diag, derivate_error_hidden_layer_transpose);

            activation_input_matrix = reLU(input_layer);
            activation_input_matrix_transpose = matrix_transpose(activation_input_matrix);
            error_weight_gradient_hidden_step = matrix_multiply(derivate_hidden_error, activation_input_matrix_transpose);
            matrix_add(error_weight_gradient_hidden, error_weight_gradient_hidden_step);

            //bias of hidden layer update
            derivate_error_hidden_layer_diag = matrix_diagonalize(derivate_error_hidden_layer_transpose);
            error_weight_gradient_bias_hidden_step = matrix_multiply(derivate_error_hidden_layer_diag, bias_hidden);
            matrix_add(error_weight_gradient_bias_hidden, error_weight_gradient_bias_hidden_step);

            matrix_multiply_constant(error_weight_gradient_bias_output_step, learning_rate);
            matrix_multiply_constant(error_weight_gradient_bias_hidden_step, learning_rate);
            matrix_multiply_constant(error_weight_gradient_output_step, learning_rate);
            matrix_multiply_constant(error_weight_gradient_hidden_step, learning_rate);
            matrix_add(bias_output, error_weight_gradient_bias_output_step);
            matrix_add(bias_hidden, error_weight_gradient_bias_hidden_step);
            matrix_add(weight_hidden_output, error_weight_gradient_output_step);
            matrix_add(weight_input_hidden, error_weight_gradient_hidden_step);
        }

        //matrix_print(weight_hidden_output);
    }
    //free block

    matrix_print(weight_hidden_output);

    for (int i = 0; i < TRAINING_SET_SIZE; i++)
    {
        //Feed forward process

        //initialisation of input layer
        input_layer->data[0][0] = training_set->data[i][0];
        input_layer->data[1][0] = training_set->data[i][1];

        //feed forward on hidden layer
        hidden_layer = matrix_multiply(weight_input_hidden, input_layer);
        matrix_add(hidden_layer, bias_hidden);
        activation_hidden_matrix = reLU(hidden_layer);

        //feed forward on output layer
        output_layer = matrix_multiply(weight_hidden_output, activation_hidden_matrix);
        matrix_add(output_layer, bias_output);
        activation_output_matrix = reLU(output_layer);
        printf("input :\n");
        matrix_print(input_layer);
        printf("output :\n");
        matrix_print(output_layer);
    }
    matrix_free(input_layer);
    matrix_free(hidden_layer);
    matrix_free(output_layer);
    matrix_free(weight_input_hidden);
    matrix_free(weight_hidden_output);
    matrix_free(bias_hidden);
    matrix_free(bias_output);
    matrix_free(training_set);
    matrix_free(expected_output);

    return 0;
}