#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "activation.h"

#define INPUT_NEURON 2
#define HIDDEN_NEURON 2
#define OUTPUT_NEURON 1
#define LAYER_NUM 3
#define TRAINING_SET_SIZE 4
#define OUPUT_SIZE 1
#define EPOCH 1000

int main()
{

    matrix_t *training_set = matrix_create(TRAINING_SET_SIZE, INPUT_NEURON);
    float training_set_data[TRAINING_SET_SIZE][INPUT_NEURON] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}};
    matrix_initize(training_set, TRAINING_SET_SIZE, INPUT_NEURON, training_set_data);

    matrix_t *expected_output = matrix_create(TRAINING_SET_SIZE, OUTPUT_NEURON);
    float expected_output_data[TRAINING_SET_SIZE][OUTPUT_NEURON] = {
        {0},
        {1},
        {1},
        {0}};
    matrix_initize(expected_output, TRAINING_SET_SIZE, OUTPUT_NEURON, expected_output_data);

    matrix_t *activation_matrix;
    matrix_t *input_layer = matrix_create(INPUT_NEURON, 1);
    matrix_t *hidden_layer;
    matrix_t *output_layer;
    matrix_t *output = matrix_create(TRAINING_SET_SIZE, OUTPUT_NEURON);
    matrix_t *derivate_output = matrix_copy(output);
    matrix_t *derivate_hidden;
    matrix_t *diagonal_error_gradient;
    matrix_t *error_weight_gradient_output = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON);
    float learning_rate = 0.1;
    matrix_t *error_weight_gradient_hidden = matrix_create(HIDDEN_NEURON, INPUT_NEURON);
    matrix_t *error_weight_gradient_output_step;
    matrix_t *error_weight_gradient_hidden_step;
    matrix_t *tmp;
    matrix_t *tmp2;
    matrix_t *tmp3;
    matrix_t *tmp4;

    matrix_t *weight_input_hidden = matrix_create(HIDDEN_NEURON, INPUT_NEURON);
    matrix_initialize_random(weight_input_hidden, 100);
    matrix_t *weight_hidden_output = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON);
    matrix_initialize_random(weight_hidden_output, 100);

    matrix_t *bias_hidden = matrix_create(HIDDEN_NEURON, 1);
    matrix_t *bias_output = matrix_create(OUTPUT_NEURON, 1);

    // XOR initialization

    bias_hidden->data[0][0] = 0;
    bias_hidden->data[1][0] = 0;

    bias_output->data[0][0] = 0;

    for (int j = 0; j < EPOCH; j++)
    {
        for (int i = 0; i < TRAINING_SET_SIZE; i++)
        {
            //Feed forward process
            input_layer->data[0][0] = training_set->data[i][0];
            input_layer->data[1][0] = training_set->data[i][1];
            hidden_layer = matrix_multiply(weight_input_hidden, input_layer);
            matrix_add(hidden_layer, bias_hidden);
            activation_matrix = reLU(hidden_layer);

            output_layer = matrix_multiply(weight_hidden_output, activation_matrix);
            matrix_add(output_layer, bias_output);
            tmp = reLU(output_layer);

            output->data[i][0] = tmp->data[0][0];

            // error function gradiant
            tmp->data[0][0] = tmp->data[0][0] - expected_output->data[i][0];

            diagonal_error_gradient = matrix_diagonalize(tmp);

            matrix_free(tmp);
            tmp = reLU_derivate(output_layer);

            derivate_hidden = matrix_multiply(tmp, diagonal_error_gradient);
            matrix_free(tmp);
            tmp2 = matrix_multiply(derivate_hidden, weight_hidden_output);
            tmp = matrix_transpose(tmp2);
            matrix_free(tmp2);
            tmp2 = reLU_derivate(output_layer);
            tmp3 = matrix_diagonalize(tmp2);
            matrix_free(tmp2);
            tmp2 = matrix_multiply(diagonal_error_gradient, tmp3);
            tmp4 = matrix_multiply(activation_matrix, tmp2);
            error_weight_gradient_output_step = matrix_transpose(tmp4);
            matrix_free(tmp2);
            matrix_free(tmp3);
            matrix_add(error_weight_gradient_output, error_weight_gradient_output_step);

            tmp2 = reLU_derivate(hidden_layer);
            tmp3 = matrix_diagonalize(tmp2);
            matrix_free(tmp2);
            tmp2 = matrix_multiply(tmp3, tmp);
            matrix_free(tmp3);
            tmp3 = unit_step(input_layer);
            tmp = matrix_transpose(tmp3);
            error_weight_gradient_hidden_step = matrix_multiply(tmp2, tmp);

            matrix_add(error_weight_gradient_hidden, error_weight_gradient_hidden_step);
        }

        matrix_multiply_constant(error_weight_gradient_output, (1.0 / TRAINING_SET_SIZE));
        matrix_multiply_constant(error_weight_gradient_output, -learning_rate);

        matrix_multiply_constant(error_weight_gradient_hidden, (1.0 / TRAINING_SET_SIZE));
        matrix_multiply_constant(error_weight_gradient_hidden, -learning_rate);
        matrix_add(weight_hidden_output, error_weight_gradient_output);
        matrix_add(weight_input_hidden, error_weight_gradient_hidden);
    }
    //free block

    matrix_print(weight_hidden_output);

    for (int i = 0; i < TRAINING_SET_SIZE; i++)
    {
        //Feed forward process
        input_layer->data[0][0] = training_set->data[i][0];
        input_layer->data[1][0] = training_set->data[i][1];
        hidden_layer = matrix_multiply(weight_input_hidden, input_layer);
        matrix_add(hidden_layer, bias_hidden);
        activation_matrix = reLU(hidden_layer);

        output_layer = matrix_multiply(weight_hidden_output, activation_matrix);
        matrix_add(output_layer, bias_output);
        tmp = reLU(output_layer);
        printf("input :\n");
        matrix_print(input_layer);
        printf("output :\n");
        matrix_print(tmp);
    }
    matrix_free(activation_matrix);
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