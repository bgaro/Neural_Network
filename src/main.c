#include <stdio.h>
#include <stdlib.h>
#include "activation.h"

#define INPUT_NEURON 2
#define HIDDEN_NEURON 2
#define OUTPUT_NEURON 1

int main()
{

    float input_layer[INPUT_NEURON] = {1, 0};
    float hidden_layer[HIDDEN_NEURON];
    float output_layer[OUTPUT_NEURON];

    float weight_input_hidden[INPUT_NEURON][HIDDEN_NEURON];
    float weight_hidden_output[HIDDEN_NEURON][OUTPUT_NEURON];

    float bias_hidden[HIDDEN_NEURON];
    float bias_output[OUTPUT_NEURON];
    float input_bias_hidden[HIDDEN_NEURON];
    float input_bias_output[OUTPUT_NEURON];

    // XOR initialization
    weight_input_hidden[0][0] = 2;
    weight_input_hidden[0][1] = -2;

    weight_input_hidden[1][0] = 2;
    weight_input_hidden[1][1] = -2;

    bias_hidden[0] = -1;
    bias_hidden[1] = 3;
    input_bias_hidden[0] = 1;
    input_bias_hidden[1] = 1;

    weight_hidden_output[0][0] = 1;
    weight_hidden_output[1][0] = 1;

    bias_output[0] = -2;
    input_bias_output[0] = 1;

    //Feed forward process
    for (int i = 0; i < HIDDEN_NEURON; i++)
    {
        hidden_layer[i] = bias_hidden[i] * input_bias_hidden[i];

        for (int j = 0; j < INPUT_NEURON; j++)
        {
            hidden_layer[i] += weight_input_hidden[j][i] * input_layer[j];
        }

        hidden_layer[i] = unit_step(hidden_layer[i]);
    }

    for (int i = 0; i < OUTPUT_NEURON; i++)
    {
        output_layer[i] = bias_output[i] * input_bias_output[i];

        for (int j = 0; j < HIDDEN_NEURON; j++)
        {
            output_layer[i] += weight_hidden_output[j][i] * hidden_layer[j];
        }

        output_layer[i] = unit_step(output_layer[i]);
    }

    printf("%f\n", output_layer[0]);

    return 0;
}