#include <stdio.h>
#include "matrix.h"
#include "activation.h"

#define SOFTMAX 0
#define RELU 1

// Feed forward

void feed_forward(matrix_t *weights, matrix_t *input, matrix_t *bias, matrix_t *output, matrix_t *activation_output, int activation)
{
    if (weights->cols != input->rows)
    {
        printf("Error: feed_forward: weights->cols != input->rows\n");
        return;
    }
    if (output->rows != bias->rows)
    {
        printf("Error: feed_forward: output->rows != bias->rows\n");
        return;
    }
    if (output->rows != weights->rows || output->cols != input->cols)
    {
        printf("Error: feed_forward: output->rows != weights->rows || output->cols != input->cols\n");
        return;
    }

    matrix_multiply(weights, input, output);
    if (activation == SOFTMAX)
    {
        matrix_add(output, bias);
        softmax(output, activation_output);
    }
    else if (activation == RELU)
    {
        int i;
        for (i = 0; i < output->rows; i++)
        {
            output->data[i] += bias->data[i];
            activation_output->data[i] = output->data[i] > 0 ? output->data[i] : 0;
        }
    }
    else
    {
        printf("Error: feed_forward: activation not supported\n");
        return;
    }
}

void backward_propagation_neurons(matrix_t *derivate_error, matrix_t *derivate_activation, matrix_t *derivate_error_activation, matrix_t *derivate_error_activation_transpose, matrix_t *weights, matrix_t *derivate_error_output_transpose, matrix_t *derivate_error_output, int activation)
{
    if (activation == RELU)
        matrix_hadamard(derivate_activation, derivate_error, derivate_error_activation);
    else if (activation == SOFTMAX)
        matrix_multiply(derivate_activation, derivate_error, derivate_error_activation);
    matrix_transpose(derivate_error_activation, derivate_error_activation_transpose);
    matrix_multiply(derivate_error_activation_transpose, weights, derivate_error_output_transpose);
    matrix_transpose(derivate_error_output_transpose, derivate_error_output);
}

void backward_propagation_weights(matrix_t *derivate_error, matrix_t *derivate_activation, matrix_t *derivate_error_activation, matrix_t *activation_layer, matrix_t *activation_layer_transpose, matrix_t *weight_derivate_output, int activation)
{
    if (activation == RELU)
        matrix_hadamard(derivate_error, derivate_activation, derivate_error_activation);
    else if (activation == SOFTMAX)
        matrix_multiply(derivate_activation, derivate_error, derivate_error_activation);
    matrix_transpose(activation_layer, activation_layer_transpose);
    matrix_multiply(derivate_error_activation, activation_layer_transpose, weight_derivate_output);
}