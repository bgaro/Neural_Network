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

int main()
{

    // [[0,0],0]
    // [[0,1],1]
    // [[1,0],1]
    // [[1,1],0]

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

    float output[4][1];

    matrix_t *input_layer = matrix_create(INPUT_NEURON, 1);
    matrix_t *hidden_layer = matrix_create(HIDDEN_NEURON, 1);
    matrix_t *output_layer = matrix_create(OUTPUT_NEURON, 1);

    float input_layer_data[INPUT_NEURON][1] = {
        {0}, {1}};
    matrix_initize(input_layer, INPUT_NEURON, 1, input_layer_data);

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
    hidden_layer = matrix_multiply(weight_input_hidden, input_layer);
    matrix_add(hidden_layer, bias_hidden);
    unit_step(hidden_layer);

    output_layer = matrix_multiply(weight_hidden_output, hidden_layer);
    matrix_add(output_layer, bias_output);
    unit_step(output_layer);

    matrix_print(output_layer);

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