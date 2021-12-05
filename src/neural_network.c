#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "matrix.h"
#include "activation.h"

#define SOFTMAX 0
#define RELU 1
typedef struct
{
    matrix_t *a;
    matrix_t *b;
    matrix_t *c;
    int index;
    int iteration;
    int max_thread;
} args;

// Feed forward

void feed_forward(matrix_t *weights, matrix_t *input, matrix_t *bias, matrix_t *output, matrix_t *activation_output, void (*activation_function)(matrix_t *, matrix_t *), int max_thread, pthread_t thread[max_thread], args *argument[max_thread])
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
    for (int k = 0; k < max_thread; k++)
    {
        argument[k]->a = weights;
        argument[k]->b = input;
        argument[k]->c = output;
        argument[k]->index = k;
        argument[k]->iteration = 0;
    }

    for (int j = 0; j < (weights->rows / max_thread) + 1; j++)
    {

        for (int i = 0; i < max_thread; i++)
        {
            pthread_create(&thread[i], NULL, matrix_multiply, argument[i]);
        }
        for (int i = 0; i < max_thread; i++)
        {
            pthread_join(thread[i], NULL);
        }
    }
    matrix_add(output, bias);
    activation_function(output, activation_output);
}

void backward_propagation_neurons(matrix_t *derivate_error, matrix_t *derivate_activation, matrix_t *derivate_error_activation, matrix_t *derivate_error_activation_transpose, matrix_t *weights, matrix_t *derivate_error_output_transpose, matrix_t *derivate_error_output, int activation, int max_thread, pthread_t thread[max_thread], args *argument[max_thread])
{
    if (activation == RELU)
        matrix_hadamard(derivate_activation, derivate_error, derivate_error_activation);
    else if (activation == SOFTMAX)
    {
        for (int k = 0; k < max_thread; k++)
        {
            argument[k]->a = derivate_activation;
            argument[k]->b = derivate_error;
            argument[k]->c = derivate_error_activation;
            argument[k]->index = k;
            argument[k]->iteration = 0;
        }
        for (int j = 0; j < (derivate_activation->rows / max_thread) + 1; j++)
        {

            for (int i = 0; i < max_thread; i++)
            {
                pthread_create(&thread[i], NULL, matrix_multiply, argument[i]);
            }
            for (int i = 0; i < max_thread; i++)
            {
                pthread_join(thread[i], NULL);
            }
        }
    }
    matrix_transpose(derivate_error_activation, derivate_error_activation_transpose);
    for (int k = 0; k < max_thread; k++)
    {
        argument[k]->a = derivate_error_activation_transpose;
        argument[k]->b = weights;
        argument[k]->c = derivate_error_output_transpose;
        argument[k]->index = k;
        argument[k]->iteration = 0;
    }
    for (int j = 0; j < (derivate_error_activation_transpose->rows / max_thread) + 1; j++)
    {

        for (int i = 0; i < max_thread; i++)
        {
            pthread_create(&thread[i], NULL, matrix_multiply, argument[i]);
        }
        for (int i = 0; i < max_thread; i++)
        {
            pthread_join(thread[i], NULL);
        }
    }
    matrix_transpose(derivate_error_output_transpose, derivate_error_output);
}

void backward_propagation_weights(matrix_t *derivate_error, matrix_t *derivate_activation, matrix_t *derivate_error_activation, matrix_t *activation_layer, matrix_t *activation_layer_transpose, matrix_t *weight_derivate_output, int activation, int max_thread, pthread_t thread[max_thread], args *argument[max_thread])
{
    if (activation == RELU)

        matrix_hadamard(derivate_error, derivate_activation, derivate_error_activation);

    else if (activation == SOFTMAX)
    {
        for (int k = 0; k < max_thread; k++)
        {
            argument[k]->a = derivate_activation;
            argument[k]->b = derivate_error;
            argument[k]->c = derivate_error_activation;
            argument[k]->index = k;
            argument[k]->iteration = 0;
        }
        for (int j = 0; j < (derivate_error->rows / max_thread) + 1; j++)
        {

            for (int i = 0; i < max_thread; i++)
            {
                pthread_create(&thread[i], NULL, matrix_multiply, argument[i]);
            }
            for (int i = 0; i < max_thread; i++)
            {
                pthread_join(thread[i], NULL);
            }
        }
    }

    matrix_transpose(activation_layer, activation_layer_transpose);
    for (int k = 0; k < max_thread; k++)
    {
        argument[k]->a = derivate_error_activation;
        argument[k]->b = activation_layer_transpose;
        argument[k]->c = weight_derivate_output;
        argument[k]->index = k;
        argument[k]->iteration = 0;
    }

    for (int j = 0; j < (derivate_error_activation->rows / max_thread) + 1; j++)
    {

        for (int i = 0; i < max_thread; i++)
        {
            pthread_create(&thread[i], NULL, matrix_multiply, argument[i]);
        }
        for (int i = 0; i < max_thread; i++)
        {
            pthread_join(thread[i], NULL);
        }
    }
}