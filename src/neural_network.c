#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "matrix.h"
#include "activation.h"

#define SOFTMAX 0
#define RELU 1
#define INPUT_NEURON 784
#define TRAINING_SET_SIZE 50

typedef struct
{
    int index;
    int thread_num;

    matrix_t *input_layer_transpose;
    float **input_array;
    float **expected_output_array;
    matrix_t *input_layer;
    matrix_t *weight_input_hidden;
    matrix_t *bias_hidden_1;
    matrix_t *hidden_layer_1;
    matrix_t *activation_hidden_1_matrix;
    matrix_t *weight_hidden_hidden;
    matrix_t *bias_hidden;
    matrix_t *hidden_layer;
    matrix_t *activation_hidden_matrix;
    matrix_t *weight_hidden_output;
    matrix_t *bias_output;
    matrix_t *output_layer;
    matrix_t *activation_output_matrix;
    matrix_t *expected_output;
    matrix_t *derivate_error_output_layer;
    matrix_t *derivate_output;
    matrix_t *derivate_output_activiation;
    matrix_t *derivate_output_activiation_transpose;
    matrix_t *derivate_error_hidden_layer_transpose;
    matrix_t *derivate_error_hidden_layer;
    matrix_t *derivate_hidden;
    matrix_t *derivate_hidden_activation;
    matrix_t *derivate_hidden_activation_transpose;
    matrix_t *derivate_error_hidden_layer_1_transpose;
    matrix_t *derivate_error_hidden_layer_1;
    matrix_t *derivate_error_activation_output;
    matrix_t *activation_hidden_matrix_transpose;
    matrix_t *error_weight_gradient_output_step;
    matrix_t *derivate_hidden_error;
    matrix_t *activation_hidden_1_matrix_transpose;
    matrix_t *error_weight_gradient_hidden_step;
    matrix_t *derivate_hidden_1_activation;
    matrix_t *derivate_hidden_1_error;
    matrix_t *error_weight_gradient_hidden_1_step;
    matrix_t *error_weight_gradient_output;
    matrix_t *error_weight_gradient_hidden;
    matrix_t *error_weight_gradient_hidden_1;
    matrix_t *error_weight_gradient_bias_hidden_1;
    matrix_t *error_weight_gradient_bias_hidden;
    matrix_t *error_weight_gradient_bias_output;

} arguments;

typedef struct
{
    matrix_t *error_weight_gradient_output;
    matrix_t *error_weight_gradient_hidden;
    matrix_t *error_weight_gradient_hidden_1;
    matrix_t *error_weight_gradient_bias_hidden_1;
    matrix_t *error_weight_gradient_bias_hidden;
    matrix_t *error_weight_gradient_bias_output;
} return_s;

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

void *training_thread(void *arg)
{
    arguments *args = (arguments *)arg;
    int index = args->index;
    int thread_num = args->thread_num;

    float **input_array = malloc(sizeof(float *) * TRAINING_SET_SIZE);
    int i;
    for (i = 0; i < TRAINING_SET_SIZE; i++)
    {
        input_array[i] = malloc(sizeof(float) * INPUT_NEURON);
        memcpy(input_array[i], args->input_array[i], sizeof(float) * INPUT_NEURON);
    }

    float **expected_output_array = malloc(sizeof(float *) * TRAINING_SET_SIZE);
    for (i = 0; i < TRAINING_SET_SIZE; i++)
    {
        expected_output_array[i] = malloc(sizeof(float) * 10);
        memcpy(expected_output_array[i], args->expected_output_array[i], sizeof(float) * 10);
    }
    matrix_t *input_layer_transpose;
    matrix_t *input_layer;
    matrix_t *weight_input_hidden;
    matrix_t *bias_hidden_1;
    matrix_t *hidden_layer_1;
    matrix_t *activation_hidden_1_matrix;
    matrix_t *weight_hidden_hidden;
    matrix_t *bias_hidden;
    matrix_t *hidden_layer;
    matrix_t *activation_hidden_matrix;
    matrix_t *weight_hidden_output;
    matrix_t *bias_output;
    matrix_t *output_layer;
    matrix_t *activation_output_matrix;
    matrix_t *expected_output;
    matrix_t *derivate_error_output_layer;
    matrix_t *derivate_output;
    matrix_t *derivate_output_activiation;
    matrix_t *derivate_output_activiation_transpose;
    matrix_t *derivate_error_hidden_layer_transpose;
    matrix_t *derivate_error_hidden_layer;
    matrix_t *derivate_hidden;
    matrix_t *derivate_hidden_activation;
    matrix_t *derivate_hidden_activation_transpose;
    matrix_t *derivate_error_hidden_layer_1_transpose;
    matrix_t *derivate_error_hidden_layer_1;
    matrix_t *derivate_error_activation_output;
    matrix_t *activation_hidden_matrix_transpose;
    matrix_t *error_weight_gradient_output_step;
    matrix_t *derivate_hidden_error;
    matrix_t *activation_hidden_1_matrix_transpose;
    matrix_t *error_weight_gradient_hidden_step;
    matrix_t *derivate_hidden_1_activation;
    matrix_t *derivate_hidden_1_error;
    matrix_t *error_weight_gradient_hidden_1_step;
    matrix_t *error_weight_gradient_output;
    matrix_t *error_weight_gradient_hidden;
    matrix_t *error_weight_gradient_hidden_1;
    matrix_t *error_weight_gradient_bias_hidden_1;
    matrix_t *error_weight_gradient_bias_hidden;
    matrix_t *error_weight_gradient_bias_output;

    input_layer_transpose = matrix_copy(args->input_layer_transpose);
    input_layer = matrix_copy(args->input_layer);
    weight_input_hidden = matrix_copy(args->weight_input_hidden);
    bias_hidden_1 = matrix_copy(args->bias_hidden_1);
    hidden_layer_1 = matrix_copy(args->hidden_layer_1);
    activation_hidden_1_matrix = matrix_copy(args->activation_hidden_1_matrix);
    weight_hidden_hidden = matrix_copy(args->weight_hidden_hidden);
    bias_hidden = matrix_copy(args->bias_hidden);
    hidden_layer = matrix_copy(args->hidden_layer);
    activation_hidden_matrix = matrix_copy(args->activation_hidden_matrix);
    weight_hidden_output = matrix_copy(args->weight_hidden_output);
    bias_output = matrix_copy(args->bias_output);
    output_layer = matrix_copy(args->output_layer);
    activation_output_matrix = matrix_copy(args->activation_output_matrix);
    expected_output = matrix_copy(args->expected_output);
    derivate_error_output_layer = matrix_copy(args->derivate_error_output_layer);
    derivate_output = matrix_copy(args->derivate_output);
    derivate_output_activiation = matrix_copy(args->derivate_output_activiation);
    derivate_output_activiation_transpose = matrix_copy(args->derivate_output_activiation_transpose);
    derivate_error_hidden_layer_transpose = matrix_copy(args->derivate_error_hidden_layer_transpose);
    derivate_error_hidden_layer = matrix_copy(args->derivate_error_hidden_layer);
    derivate_hidden = matrix_copy(args->derivate_hidden);
    derivate_hidden_activation = matrix_copy(args->derivate_hidden_activation);
    derivate_hidden_activation_transpose = matrix_copy(args->derivate_hidden_activation_transpose);
    derivate_error_hidden_layer_1_transpose = matrix_copy(args->derivate_error_hidden_layer_1_transpose);
    derivate_error_hidden_layer_1 = matrix_copy(args->derivate_error_hidden_layer_1);
    derivate_error_activation_output = matrix_copy(args->derivate_error_activation_output);
    activation_hidden_matrix_transpose = matrix_copy(args->activation_hidden_matrix_transpose);
    error_weight_gradient_output_step = matrix_copy(args->error_weight_gradient_output_step);
    derivate_hidden_error = matrix_copy(args->derivate_hidden_error);
    activation_hidden_1_matrix_transpose = matrix_copy(args->activation_hidden_1_matrix_transpose);
    error_weight_gradient_hidden_step = matrix_copy(args->error_weight_gradient_hidden_step);
    derivate_hidden_1_activation = matrix_copy(args->derivate_hidden_1_activation);
    derivate_hidden_1_error = matrix_copy(args->derivate_hidden_1_error);
    error_weight_gradient_hidden_1_step = matrix_copy(args->error_weight_gradient_hidden_1_step);
    error_weight_gradient_output = matrix_copy(args->error_weight_gradient_output);
    error_weight_gradient_hidden = matrix_copy(args->error_weight_gradient_hidden);
    error_weight_gradient_hidden_1 = matrix_copy(args->error_weight_gradient_hidden_1);
    error_weight_gradient_bias_hidden_1 = matrix_copy(args->error_weight_gradient_bias_hidden_1);
    error_weight_gradient_bias_hidden = matrix_copy(args->error_weight_gradient_bias_hidden);
    error_weight_gradient_bias_output = matrix_copy(args->error_weight_gradient_bias_output);

    for (i = index; i < TRAINING_SET_SIZE;)
    {
        // Feed forward process

        // initialisation of input layer
        // printf("%d : ", i);
        matrix_initialize(input_layer_transpose, 1, INPUT_NEURON, input_array[i]);
        matrix_transpose(input_layer_transpose, input_layer);

        // feed forward on hidden layer 1
        feed_forward(weight_input_hidden, input_layer, bias_hidden_1, hidden_layer_1, activation_hidden_1_matrix, RELU);

        // feed forward on hidden layer
        feed_forward(weight_hidden_hidden, activation_hidden_1_matrix, bias_hidden, hidden_layer, activation_hidden_matrix, RELU);

        // feed forward on output layer
        feed_forward(weight_hidden_output, activation_hidden_matrix, bias_output, output_layer, activation_output_matrix, SOFTMAX);

        // error function gradiant

        // yj - dkj for j in Y (output layer)
        // matrix_initialize(expected_output_step, expected_output_step->rows, expected_output_step->cols, &expected_output->data[i][0]);

        matrix_initialize(expected_output, expected_output->rows, expected_output->cols, expected_output_array[i]);
        matrix_transpose(expected_output, derivate_error_output_layer);
        matrix_subtract(derivate_error_output_layer, activation_output_matrix);
        matrix_multiply_constant(derivate_error_output_layer, -(1.0 / TRAINING_SET_SIZE));
        softmax_derivate(activation_output_matrix, derivate_output);

        /*for (int d = 0; d < OUTPUT_NEURON; d++)
        {
            error += expected_output_array[0][d] * log(activation_output_matrix->data[d]);
        }*/

        // dEk/dyj for j in Y (output layer)

        // dEk/dyj for j in Z \ (Y U X) (hidden layer)
        backward_propagation_neurons(derivate_error_output_layer, derivate_output, derivate_output_activiation, derivate_output_activiation_transpose, weight_hidden_output, derivate_error_hidden_layer_transpose, derivate_error_hidden_layer, SOFTMAX);

        // dEk/dwij for j in Z \ (Y U X) (hidden layer) 3
        reLU_derivate(hidden_layer, derivate_hidden);
        backward_propagation_neurons(derivate_error_hidden_layer, derivate_hidden, derivate_hidden_activation, derivate_hidden_activation_transpose, weight_hidden_hidden, derivate_error_hidden_layer_1_transpose, derivate_error_hidden_layer_1, RELU);

        // dEk/dwij for j in Y(output layer)

        backward_propagation_weights(derivate_error_output_layer, derivate_output, derivate_error_activation_output, activation_hidden_matrix, activation_hidden_matrix_transpose, error_weight_gradient_output_step, SOFTMAX);

        // dEk/dwij for j in Z \ (Y U X) (hidden layer) 4

        backward_propagation_weights(derivate_error_hidden_layer, derivate_hidden, derivate_hidden_error, activation_hidden_1_matrix, activation_hidden_1_matrix_transpose, error_weight_gradient_hidden_step, RELU);

        // dEk/dwij for j in Z \ (Y U X) (hidden layer) 3

        reLU_derivate(hidden_layer_1, derivate_hidden_1_activation);
        backward_propagation_weights(derivate_error_hidden_layer_1, derivate_hidden_1_activation, derivate_hidden_1_error, input_layer, input_layer_transpose, error_weight_gradient_hidden_1_step, RELU);

        matrix_add(error_weight_gradient_output, error_weight_gradient_output_step); // sum of each training set
        matrix_add(error_weight_gradient_hidden, error_weight_gradient_hidden_step);
        matrix_add(error_weight_gradient_hidden_1, error_weight_gradient_hidden_1_step);

        // bias of hidden layer 3 update
        matrix_add(error_weight_gradient_bias_hidden_1, derivate_hidden_1_error);

        // bias of hidden layer 4 update
        matrix_add(error_weight_gradient_bias_hidden, derivate_hidden_error);
        // bias of output layer update

        matrix_add(error_weight_gradient_bias_output, derivate_error_activation_output);
        i += thread_num;
    }
    return_s *return_struct = malloc(sizeof(return_s));
    return_struct->error_weight_gradient_output = matrix_copy(error_weight_gradient_output);
    return_struct->error_weight_gradient_hidden = matrix_copy(error_weight_gradient_hidden);
    return_struct->error_weight_gradient_hidden_1 = matrix_copy(error_weight_gradient_hidden_1);
    return_struct->error_weight_gradient_bias_hidden_1 = matrix_copy(error_weight_gradient_bias_hidden_1);
    return_struct->error_weight_gradient_bias_hidden = matrix_copy(error_weight_gradient_bias_hidden);
    return_struct->error_weight_gradient_bias_output = matrix_copy(error_weight_gradient_bias_output);
    matrix_free(input_layer_transpose);
    matrix_free(input_layer);
    matrix_free(weight_input_hidden);
    matrix_free(bias_hidden_1);
    matrix_free(hidden_layer_1);
    matrix_free(activation_hidden_1_matrix);
    matrix_free(weight_hidden_hidden);
    matrix_free(bias_hidden);
    matrix_free(hidden_layer);
    matrix_free(activation_hidden_matrix);
    matrix_free(weight_hidden_output);
    matrix_free(bias_output);
    matrix_free(output_layer);
    matrix_free(activation_output_matrix);
    matrix_free(expected_output);
    matrix_free(derivate_error_output_layer);
    matrix_free(derivate_output);
    matrix_free(derivate_output_activiation);
    matrix_free(derivate_output_activiation_transpose);
    matrix_free(derivate_error_hidden_layer_transpose);
    matrix_free(derivate_error_hidden_layer);
    matrix_free(derivate_hidden);
    matrix_free(derivate_hidden_activation);
    matrix_free(derivate_hidden_activation_transpose);
    matrix_free(derivate_error_hidden_layer_1_transpose);
    matrix_free(derivate_error_hidden_layer_1);
    matrix_free(derivate_error_activation_output);
    matrix_free(activation_hidden_matrix_transpose);
    matrix_free(error_weight_gradient_output_step);
    matrix_free(derivate_hidden_error);
    matrix_free(activation_hidden_1_matrix_transpose);
    matrix_free(error_weight_gradient_hidden_step);
    matrix_free(derivate_hidden_1_activation);
    matrix_free(derivate_hidden_1_error);
    matrix_free(error_weight_gradient_hidden_1_step);
    matrix_free(error_weight_gradient_output);
    matrix_free(error_weight_gradient_hidden);
    matrix_free(error_weight_gradient_hidden_1);
    matrix_free(error_weight_gradient_bias_hidden_1);
    matrix_free(error_weight_gradient_bias_hidden);
    matrix_free(error_weight_gradient_bias_output);

    for (i = 0; i < TRAINING_SET_SIZE; i++)
    {
        free(input_array[i]);
        free(expected_output_array[i]);
    }
    free(input_array);
    free(expected_output_array);

    return (void *)return_struct;
}