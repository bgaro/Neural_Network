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
    matrix_t *tmp;

    matrix_t *weight_input_hidden = matrix_create(HIDDEN_NEURON, INPUT_NEURON);
    matrix_t *weight_hidden_output = matrix_create(OUTPUT_NEURON, HIDDEN_NEURON);

    matrix_t *bias_hidden = matrix_create(HIDDEN_NEURON, 1);
    matrix_t *bias_output = matrix_create(OUTPUT_NEURON, 1);

    // XOR initialization
    weight_input_hidden->data[0][0] = 2;
    weight_input_hidden->data[1][0] = -2;

    weight_input_hidden->data[0][1] = 2;
    weight_input_hidden->data[1][1] = -2;

    bias_hidden->data[0][0] = -1;
    bias_hidden->data[1][0] = 3;

    weight_hidden_output->data[0][0] = 1;
    weight_hidden_output->data[0][1] = 1;

    bias_output->data[0][0] = -2;

    //Feed forward process
    for (int i = 0; i < TRAINING_SET_SIZE; i++)
    {
        input_layer->data[0][0] = training_set->data[i][0];
        input_layer->data[1][0] = training_set->data[i][1];
        hidden_layer = matrix_multiply(weight_input_hidden, input_layer);
        matrix_add(hidden_layer, bias_hidden);
        activation_matrix = unit_step(hidden_layer);

        output_layer = matrix_multiply(weight_hidden_output, activation_matrix);
        matrix_add(output_layer, bias_output);
        matrix_free(activation_matrix);
        tmp = unit_step(output_layer);
        output->data[i][0] = tmp->data[0][0];

        // error function gradiant
        derivate_output->data[i][0] = tmp->data[0][0] - expected_output->data[i][0];
        matrix_free(tmp);

        tmp = matrix_create(1, OUPUT_SIZE);
        for (int j = 0; j < OUPUT_SIZE; j++)
        {
            tmp->data[0][j] = derivate_output->data[i][0];
        }
        diagonal_error_gradient = matrix_diagonalize(tmp);
        matrix_free(tmp);
        tmp = reLU_derivate(output_layer);
        derivate_hidden = matrix_multiply(tmp, diagonal_error_gradient);
        matrix_free(tmp);
        tmp = matrix_multiply(derivate_hidden, weight_hidden_output);
        matrix_print(tmp);
    }
    //Backpropagation process
    printf("test\n");

    matrix_subtract(derivate_output, expected_output);

    matrix_print(derivate_output);

    //free block
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