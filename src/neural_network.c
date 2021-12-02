#include <stdio.h>
#include "matrix.h"
#include "activation.h"

// Feed forward

void feed_forward(matrix_t *weights, matrix_t *input, matrix_t *bias, matrix_t *output, matrix_t *activation_output, void (*activation_function)(matrix_t *, matrix_t *))
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
    matrix_add(output, bias);
    activation_function(output, activation_output);
}