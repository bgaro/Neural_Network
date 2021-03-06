#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include "matrix.h"
#include "activation.h"
#include "csv_to_array.h"
#include "neural_network.h"

#define NUM_THREADS 16
#define INPUT_NEURON 784
#define HIDDEN_NEURON_1 21
#define HIDDEN_NEURON 45
#define OUTPUT_NEURON 10
#define LAYER_NUM 4
#define TRAINING_SET_SIZE 60000
#define TEST_SET_SIZE 10000
#define EPOCH 1866
int main()
{
srand(0);
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
    remove("actualTestPredictions");
    remove("trainPredictions");
    FILE *test_vectors_stream = fopen("./data/fashion_mnist_test_vectors.csv", "r");
    FILE *test_labels_stream = fopen("./data/fashion_mnist_test_labels.csv", "r");

    float **input_array;
    float **input_array_test;
    int epoch = 0;
    int training = 0;
    int i;
    pthread_t threads[NUM_THREADS];
    arguments *args[NUM_THREADS];
    for (i = 0; i < NUM_THREADS; i++)
    {
        args[i] = malloc(sizeof(arguments));
    }
    return_s returns[NUM_THREADS];
    void *return_value[NUM_THREADS];
    time_t start, end;
    start = time(NULL);

    input_array = csv_to_array_vectors(train_vectors_stream, TRAINING_SET_SIZE);
    float **expected_output_array;
    float **expected_output_array_test;
    expected_output_array = csv_to_array_labels(train_labels_stream, TRAINING_SET_SIZE);
    expected_output_array_test = csv_to_array_labels(test_labels_stream, TEST_SET_SIZE);
    input_array_test = csv_to_array_vectors(test_vectors_stream, TEST_SET_SIZE);

    float learning_rate = 0.12;
    float alpha = 0.95;

    int train_prediction_fd = 0;
    int test_prediction_fd = 0;
    int prediction = 0;
    //int prediction_2 = 0;
   // float max = 0.0;
    //int epoch_max = 0;
    //int cpt = 0;
    char prediction_buffer[3];
    train_prediction_fd = open("trainPredictions", O_CREAT | O_WRONLY, 0644);
    test_prediction_fd = open("actualTestPredictions", O_CREAT | O_WRONLY, 0644);

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
    printf("Parameters : EPOCH : %i LEARNING RATE : %f MOMENTUM : %f\nHIDDEN_LAYER 1 : %i HIDDEN_LAYER_2 : %i TRAINING_SET_SIZE : %i\n", EPOCH, learning_rate, alpha, HIDDEN_NEURON_1, HIDDEN_NEURON, TRAINING_SET_SIZE);

    for (training = 0; training < NUM_THREADS; training++)
    {
        args[training]->index = training;
        args[training]->thread_num = NUM_THREADS;
        args[training]->input_array = malloc(sizeof(float *) * TRAINING_SET_SIZE);
        args[training]->expected_output_array = malloc(sizeof(float *) * TRAINING_SET_SIZE);

        for (i = 0; i < TRAINING_SET_SIZE; i++)
        {
            args[training]->input_array[i] = malloc(sizeof(float) * INPUT_NEURON);
            memcpy(args[training]->input_array[i], input_array[i], sizeof(float) * INPUT_NEURON);
        }

        for (i = 0; i < TRAINING_SET_SIZE; i++)
        {
            args[training]->expected_output_array[i] = malloc(sizeof(float) * OUTPUT_NEURON);
            memcpy(args[training]->expected_output_array[i], expected_output_array[i], sizeof(float) * OUTPUT_NEURON);
        }

        args[training]->input_layer_transpose = matrix_copy(input_layer_transpose);
        args[training]->input_layer = matrix_copy(input_layer);
        args[training]->weight_input_hidden = matrix_copy(weight_input_hidden);
        args[training]->bias_hidden_1 = matrix_copy(bias_hidden_1);
        args[training]->hidden_layer_1 = matrix_copy(hidden_layer_1);
        args[training]->activation_hidden_1_matrix = matrix_copy(activation_hidden_1_matrix);
        args[training]->weight_hidden_hidden = matrix_copy(weight_hidden_hidden);
        args[training]->bias_hidden = matrix_copy(bias_hidden);
        args[training]->hidden_layer = matrix_copy(hidden_layer);
        args[training]->activation_hidden_matrix = matrix_copy(activation_hidden_matrix);
        args[training]->weight_hidden_output = matrix_copy(weight_hidden_output);
        args[training]->bias_output = matrix_copy(bias_output);
        args[training]->output_layer = matrix_copy(output_layer);
        args[training]->activation_output_matrix = matrix_copy(activation_output_matrix);
        args[training]->expected_output = matrix_copy(expected_output);
        args[training]->derivate_error_output_layer = matrix_copy(derivate_error_output_layer);
        args[training]->derivate_output = matrix_copy(derivate_output);
        args[training]->derivate_output_activiation = matrix_copy(derivate_output_activiation);
        args[training]->derivate_output_activiation_transpose = matrix_copy(derivate_output_activiation_transpose);
        args[training]->derivate_error_hidden_layer_transpose = matrix_copy(derivate_error_hidden_layer_transpose);
        args[training]->derivate_error_hidden_layer = matrix_copy(derivate_error_hidden_layer);
        args[training]->derivate_hidden = matrix_copy(derivate_hidden);
        args[training]->derivate_hidden_activation = matrix_copy(derivate_hidden_activation);
        args[training]->derivate_hidden_activation_transpose = matrix_copy(derivate_hidden_activation_transpose);
        args[training]->derivate_error_hidden_layer_1_transpose = matrix_copy(derivate_error_hidden_layer_1_transpose);
        args[training]->derivate_error_hidden_layer_1 = matrix_copy(derivate_error_hidden_layer_1);
        args[training]->derivate_error_activation_output = matrix_copy(derivate_error_activation_output);
        args[training]->activation_hidden_matrix_transpose = matrix_copy(activation_hidden_matrix_transpose);
        args[training]->error_weight_gradient_output_step = matrix_copy(error_weight_gradient_output_step);
        args[training]->derivate_hidden_error = matrix_copy(derivate_hidden_error);
        args[training]->activation_hidden_1_matrix_transpose = matrix_copy(activation_hidden_1_matrix_transpose);
        args[training]->error_weight_gradient_hidden_step = matrix_copy(error_weight_gradient_hidden_step);
        args[training]->derivate_hidden_1_activation = matrix_copy(derivate_hidden_1_activation);
        args[training]->derivate_hidden_1_error = matrix_copy(derivate_hidden_1_error);
        args[training]->error_weight_gradient_hidden_1_step = matrix_copy(error_weight_gradient_hidden_1_step);
        args[training]->error_weight_gradient_output = matrix_copy(error_weight_gradient_output);
        args[training]->error_weight_gradient_hidden = matrix_copy(error_weight_gradient_hidden);
        args[training]->error_weight_gradient_hidden_1 = matrix_copy(error_weight_gradient_hidden_1);
        args[training]->error_weight_gradient_bias_hidden_1 = matrix_copy(error_weight_gradient_bias_hidden_1);
        args[training]->error_weight_gradient_bias_hidden = matrix_copy(error_weight_gradient_bias_hidden);
        args[training]->error_weight_gradient_bias_output = matrix_copy(error_weight_gradient_bias_output);
    }

    for (epoch = 0; epoch < EPOCH; epoch++)
    {
        for (training = 0; training < NUM_THREADS; training++)
        {
            pthread_create(&threads[training], NULL, training_thread, (void *)args[training]);
        }
        for (training = 0; training < NUM_THREADS; training++)
        {
            pthread_join(threads[training], &return_value[training]);
            memcpy(&returns[training], return_value[training], sizeof(return_s));
            free(return_value[training]);
        }
        for (training = 0; training < NUM_THREADS; training++)
        {

            matrix_add(error_weight_gradient_output, returns[training].error_weight_gradient_output);
            matrix_add(error_weight_gradient_hidden, returns[training].error_weight_gradient_hidden);
            matrix_add(error_weight_gradient_hidden_1, returns[training].error_weight_gradient_hidden_1);
            matrix_add(error_weight_gradient_bias_hidden, returns[training].error_weight_gradient_bias_hidden);
            matrix_add(error_weight_gradient_bias_hidden_1, returns[training].error_weight_gradient_bias_hidden_1);
            matrix_add(error_weight_gradient_bias_output, returns[training].error_weight_gradient_bias_output);

            matrix_free(returns[training].error_weight_gradient_output);
            matrix_free(returns[training].error_weight_gradient_hidden);
            matrix_free(returns[training].error_weight_gradient_hidden_1);
            matrix_free(returns[training].error_weight_gradient_bias_hidden);
            matrix_free(returns[training].error_weight_gradient_bias_hidden_1);
            matrix_free(returns[training].error_weight_gradient_bias_output);
        }

        /*for (training = 0; training < TRAINING_SET_SIZE; training++)
        {
            // Feed forward process

            // initialisation of input layer

            matrix_initialize(input_layer_transpose, 1, INPUT_NEURON, input_array[training]);
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

            matrix_initialize(expected_output, expected_output->rows, expected_output->cols, expected_output_array[training]);
            matrix_transpose(expected_output, derivate_error_output_layer);
            matrix_subtract(derivate_error_output_layer, activation_output_matrix);
            matrix_multiply_constant(derivate_error_output_layer, -(1.0 / TRAINING_SET_SIZE));
            softmax_derivate(activation_output_matrix, derivate_output);

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

            // free memory
        }*/
        if (epoch % 50 == 0)
        {
            printf("Epoch %d\n", epoch);
        }
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

        matrix_free(error_weight_gradient_bias_output_previous_step);
        matrix_free(error_weight_gradient_bias_hidden_previous_step);
        matrix_free(error_weight_gradient_bias_hidden_1_previous_step);

        error_weight_gradient_bias_output_previous_step = matrix_copy(error_weight_gradient_bias_output);
        error_weight_gradient_bias_hidden_previous_step = matrix_copy(error_weight_gradient_bias_hidden);
        error_weight_gradient_bias_hidden_1_previous_step = matrix_copy(error_weight_gradient_bias_hidden_1);

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

        matrix_free(error_weight_gradient_output_previous_step);
        matrix_free(error_weight_gradient_hidden_previous_step);
        matrix_free(error_weight_gradient_hidden_1_previous_step);

        error_weight_gradient_output_previous_step = matrix_copy(error_weight_gradient_output);
        error_weight_gradient_hidden_previous_step = matrix_copy(error_weight_gradient_hidden);
        error_weight_gradient_hidden_1_previous_step = matrix_copy(error_weight_gradient_hidden_1);

        matrix_add(weight_hidden_output, error_weight_gradient_output);
        matrix_add(weight_input_hidden, error_weight_gradient_hidden_1);
        matrix_add(weight_hidden_hidden, error_weight_gradient_hidden);

        matrix_reset(error_weight_gradient_output);
        matrix_reset(error_weight_gradient_hidden);
        matrix_reset(error_weight_gradient_hidden_1);

        for (i = 0; i < NUM_THREADS; i++)
        {
            matrix_free(args[i]->error_weight_gradient_output);
            matrix_free(args[i]->error_weight_gradient_hidden);
            matrix_free(args[i]->error_weight_gradient_hidden_1);
            matrix_free(args[i]->error_weight_gradient_bias_output);
            matrix_free(args[i]->error_weight_gradient_bias_hidden);
            matrix_free(args[i]->error_weight_gradient_bias_hidden_1);
            matrix_free(args[i]->weight_hidden_hidden);
            matrix_free(args[i]->weight_input_hidden);
            matrix_free(args[i]->weight_hidden_output);
            matrix_free(args[i]->bias_hidden);
            matrix_free(args[i]->bias_hidden_1);
            matrix_free(args[i]->bias_output);

            args[i]->error_weight_gradient_bias_output = matrix_copy(error_weight_gradient_bias_output);
            args[i]->error_weight_gradient_bias_hidden = matrix_copy(error_weight_gradient_bias_hidden);
            args[i]->error_weight_gradient_bias_hidden_1 = matrix_copy(error_weight_gradient_bias_hidden_1);
            args[i]->error_weight_gradient_output = matrix_copy(error_weight_gradient_output);
            args[i]->error_weight_gradient_hidden = matrix_copy(error_weight_gradient_hidden);
            args[i]->error_weight_gradient_hidden_1 = matrix_copy(error_weight_gradient_hidden_1);
            args[i]->weight_hidden_output = matrix_copy(weight_hidden_output);
            args[i]->weight_input_hidden = matrix_copy(weight_input_hidden);
            args[i]->weight_hidden_hidden = matrix_copy(weight_hidden_hidden);
            args[i]->bias_output = matrix_copy(bias_output);
            args[i]->bias_hidden = matrix_copy(bias_hidden);
            args[i]->bias_hidden_1 = matrix_copy(bias_hidden_1);
        }

/*        for (training = 0; training < TEST_SET_SIZE; training++)
        {

            // Feed forward process

            // initialisation of input layer

            matrix_initialize(input_layer_transpose, 1, INPUT_NEURON, input_array_test[training]);
            matrix_transpose(input_layer_transpose, input_layer);

            // feed forward on hidden layer 1
            feed_forward(weight_input_hidden, input_layer, bias_hidden_1, hidden_layer_1, activation_hidden_1_matrix, RELU);

            // feed forward on hidden layer
            feed_forward(weight_hidden_hidden, activation_hidden_1_matrix, bias_hidden, hidden_layer, activation_hidden_matrix, RELU);

            // feed forward on output layer
            feed_forward(weight_hidden_output, activation_hidden_matrix, bias_output, output_layer, activation_output_matrix, SOFTMAX);
            // error function gradiant
            matrix_initialize(expected_output, expected_output->rows, expected_output->cols, expected_output_array_test[training]);
            prediction = get_label(activation_output_matrix);
            prediction_2 = get_label(expected_output);
            if (prediction == prediction_2)
                cpt++;
        }

        printf("EPOCH : %d | ACCURACY : %f \n", epoch, (float)cpt / (float)TEST_SET_SIZE * 100);
        if (((float)cpt / (float)TEST_SET_SIZE * 100) > max)
        {
            max = (float)cpt / (float)TEST_SET_SIZE * 100;
            epoch_max = epoch;
        }
        cpt = 0;
*/
        // reset file pointer
    }
    //printf("best : EPOCH : %d | ACCURACY : %f \n", epoch_max, max);
    for (training = 0; training < TRAINING_SET_SIZE; training++)
    {

        // Feed forward process

        // initialisation of input layer

        matrix_initialize(input_layer_transpose, 1, INPUT_NEURON, input_array[training]);
        matrix_transpose(input_layer_transpose, input_layer);

        // feed forward on hidden layer 1
        feed_forward(weight_input_hidden, input_layer, bias_hidden_1, hidden_layer_1, activation_hidden_1_matrix, RELU);

        // feed forward on hidden layer
        feed_forward(weight_hidden_hidden, activation_hidden_1_matrix, bias_hidden, hidden_layer, activation_hidden_matrix, RELU);

        // feed forward on output layer
        feed_forward(weight_hidden_output, activation_hidden_matrix, bias_output, output_layer, activation_output_matrix, SOFTMAX);
        // error function gradiant
        prediction = get_label(activation_output_matrix);
        sprintf(prediction_buffer, "%d\n", prediction);
        if ((write(train_prediction_fd, prediction_buffer, strlen(prediction_buffer))) < 0)
        {
            printf("Error writing in a file\n");
            return 1;
        }
    }

    for (training = 0; training < NUM_THREADS; training++)
    {
        matrix_free(args[training]->input_layer_transpose);
        matrix_free(args[training]->input_layer);
        matrix_free(args[training]->weight_input_hidden);
        matrix_free(args[training]->bias_hidden_1);
        matrix_free(args[training]->hidden_layer_1);
        matrix_free(args[training]->activation_hidden_1_matrix);
        matrix_free(args[training]->weight_hidden_hidden);
        matrix_free(args[training]->bias_hidden);
        matrix_free(args[training]->hidden_layer);
        matrix_free(args[training]->activation_hidden_matrix);
        matrix_free(args[training]->weight_hidden_output);
        matrix_free(args[training]->bias_output);
        matrix_free(args[training]->output_layer);
        matrix_free(args[training]->activation_output_matrix);
        matrix_free(args[training]->expected_output);
        matrix_free(args[training]->derivate_error_output_layer);
        matrix_free(args[training]->derivate_output);
        matrix_free(args[training]->derivate_output_activiation);
        matrix_free(args[training]->derivate_output_activiation_transpose);
        matrix_free(args[training]->derivate_error_hidden_layer_transpose);
        matrix_free(args[training]->derivate_error_hidden_layer);
        matrix_free(args[training]->derivate_hidden);
        matrix_free(args[training]->derivate_hidden_activation);
        matrix_free(args[training]->derivate_hidden_activation_transpose);
        matrix_free(args[training]->derivate_error_hidden_layer_1_transpose);
        matrix_free(args[training]->derivate_error_hidden_layer_1);
        matrix_free(args[training]->derivate_error_activation_output);
        matrix_free(args[training]->activation_hidden_matrix_transpose);
        matrix_free(args[training]->error_weight_gradient_output_step);
        matrix_free(args[training]->derivate_hidden_error);
        matrix_free(args[training]->activation_hidden_1_matrix_transpose);
        matrix_free(args[training]->error_weight_gradient_hidden_step);
        matrix_free(args[training]->derivate_hidden_1_activation);
        matrix_free(args[training]->derivate_hidden_1_error);
        matrix_free(args[training]->error_weight_gradient_hidden_1_step);
        matrix_free(args[training]->error_weight_gradient_output);
        matrix_free(args[training]->error_weight_gradient_hidden);
        matrix_free(args[training]->error_weight_gradient_hidden_1);
        matrix_free(args[training]->error_weight_gradient_bias_hidden_1);
        matrix_free(args[training]->error_weight_gradient_bias_hidden);
        matrix_free(args[training]->error_weight_gradient_bias_output);
    }
    for (i = 0; i < TRAINING_SET_SIZE; i++)
    {
        free(input_array[i]);
        free(expected_output_array[i]);
    }
    free(input_array);
    free(expected_output_array);

    for (training = 0; training < NUM_THREADS; training++)
    {
        for (i = 0; i < TRAINING_SET_SIZE; i++)
        {
            free(args[training]->expected_output_array[i]);
            free(args[training]->input_array[i]);
        }
        free(args[training]->expected_output_array);
        free(args[training]->input_array);
        free(args[training]);
    }

    for (training = 0; training < TEST_SET_SIZE; training++)
    {

        // Feed forward process

        // initialisation of input layer

        matrix_initialize(input_layer_transpose, 1, INPUT_NEURON, input_array_test[training]);
        matrix_transpose(input_layer_transpose, input_layer);

        // feed forward on hidden layer 1
        feed_forward(weight_input_hidden, input_layer, bias_hidden_1, hidden_layer_1, activation_hidden_1_matrix, RELU);

        // feed forward on hidden layer
        feed_forward(weight_hidden_hidden, activation_hidden_1_matrix, bias_hidden, hidden_layer, activation_hidden_matrix, RELU);

        // feed forward on output layer
        feed_forward(weight_hidden_output, activation_hidden_matrix, bias_output, output_layer, activation_output_matrix, SOFTMAX);
        // error function gradiant
        prediction = get_label(activation_output_matrix);
        sprintf(prediction_buffer, "%d\n", prediction);
        if ((write(test_prediction_fd, prediction_buffer, strlen(prediction_buffer))) < 0)
        {
            printf("Error writing in a file\n");
            return 1;
        }
    }
    printf("Parameters : EPOCH : %i LEARNING RATE : %f MOMENTUM : %f\nHIDDEN_LAYER 1 : %i HIDDEN_LAYER_2 : %i TRAINING_SET_SIZE : %i\n", EPOCH, learning_rate, alpha, HIDDEN_NEURON_1, HIDDEN_NEURON, TRAINING_SET_SIZE);
    // free block
    end = time(NULL);
    printf("Time : %f\n", (double)(end - start) / 60.0);
    for (training = 0; training < TEST_SET_SIZE; training++)
    {
        free(expected_output_array_test[training]);
        free(input_array_test[training]);
    }
    free(expected_output_array_test);
    free(input_array_test);

    matrix_free(input_layer);
    matrix_free(input_layer_transpose);
    matrix_free(hidden_layer);
    matrix_free(hidden_layer_1);
    matrix_free(activation_hidden_matrix);
    matrix_free(output_layer);
    matrix_free(activation_output_matrix);
    matrix_free(derivate_error_output_layer);
    matrix_free(derivate_hidden_1_activation);
    matrix_free(derivate_error_output_layer_diag);
    matrix_free(derivate_output);
    matrix_free(derivate_output_diag);
    matrix_free(derivate_output_activiation);
    matrix_free(error_weight_gradient_output);
    matrix_free(error_weight_gradient_bias_output);
    matrix_free(error_weight_gradient_output_step);
    matrix_free(error_weight_gradient_bias_output_step);
    matrix_free(activation_input_matrix);
    matrix_free(error_weight_gradient_hidden_1_step);
    matrix_free(error_weight_gradient_output_step_transpose);
    matrix_free(activation_input_matrix_transpose);
    matrix_free(bias_hidden);
    matrix_free(bias_hidden_1);
    matrix_free(bias_output);
    matrix_free(weight_hidden_hidden);
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
    matrix_free(error_weight_gradient_hidden_1_previous_step);
    matrix_free(derivate_hidden_error);
    matrix_free(derivate_error_activation_output);
    matrix_free(error_weight_gradient_bias_output_previous_step);
    matrix_free(error_weight_gradient_bias_hidden_previous_step);
    matrix_free(error_weight_gradient_output_previous_step);
    matrix_free(error_weight_gradient_hidden_previous_step);
    matrix_free(activation_hidden_1_matrix);
    matrix_free(activation_hidden_1_matrix_transpose);
    matrix_free(derivate_error_hidden_layer_1_transpose);
    matrix_free(derivate_error_hidden_layer_1);
    matrix_free(error_weight_gradient_bias_hidden_1);
    matrix_free(error_weight_gradient_bias_hidden_1_previous_step);
    matrix_free(derivate_hidden_1_error);
    matrix_free(error_weight_gradient_hidden_1);
    matrix_free(derivate_hidden_activation_transpose);
    return 0;
}
