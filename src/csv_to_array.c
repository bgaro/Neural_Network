#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

#define MAX_TRAIN_VECTORS_LENGTH 2968 // Tests have been made
#define IMAGE_SIZE 784                // Images 28*28

float **csv_to_array_vectors(FILE *train_vectors_stream)
{

    /* Initializations */
    int column = 0;
    char *train_vectors;

    char *train_vectors_string;
    train_vectors_string = malloc(sizeof(char) * MAX_TRAIN_VECTORS_LENGTH);
    float **train_vectors_array;
    train_vectors_array = malloc(sizeof(float *));
    train_vectors_array[0] = calloc(sizeof(float), IMAGE_SIZE);

    /* Reading the whole csv line by line */
    if ((fgets(train_vectors_string, MAX_TRAIN_VECTORS_LENGTH, train_vectors_stream)) != NULL)
    {

        train_vectors = strtok(train_vectors_string, ",");

        /* Adding properly the vector to the array of all vectors */
        while (train_vectors != NULL)
        {
            train_vectors_array[0][column] = (float)atoi(train_vectors) / 255.0;
            train_vectors = strtok(NULL, ",");
            column += 1;
        }

        /* Reset values to read next ones */
        memset(train_vectors_string, 0, MAX_TRAIN_VECTORS_LENGTH);
        column = 0;

        /* Print pictures */
        /*printf("***** IMAGE %i *****\n\n", cpt);

        for (int i = 0; i < IMAGE_SIZE; i++)
        {
            if (i % 28 == 0 && i != 0)
                printf("\n");
            printf("%-4.0f", train_vectors_array[0][i]);
        }
        printf("\n\n\n\n\n");
        cpt += 1;*/
        free(train_vectors_string);
        fseek(train_vectors_stream, 1, SEEK_CUR);
        return train_vectors_array;
    }
    return NULL;
}

int get_label(matrix_t *labels)
{
    float max = labels->data[0];
    int index;
    for (int i = 0; i < labels->rows * labels->cols; i++)
    {
        if (labels->data[i] > max)
        {
            max = labels->data[i];
            index = i;
        }
    }
    return index;
}
int csv_to_array_labels_int(FILE *train_vectors_stream)
{

    /* Initializations */
    int value;

    /* Getting the class of each vector */
    if ((value = fgetc(train_vectors_stream)) != EOF)
    {
        value = value - 48;

        fseek(train_vectors_stream, 2, SEEK_CUR);

        return value;
    }
    return -1;
}

float **csv_to_array_labels(FILE *train_vectors_stream)
{

    /* Initializations */
    int value;
    float **train_vectors_array;
    train_vectors_array = malloc(sizeof(float *));
    train_vectors_array[0] = calloc(sizeof(float), 10);

    /* Getting the class of each vector */
    if ((value = fgetc(train_vectors_stream)) != EOF)
    {
        value = value - 48;

        fseek(train_vectors_stream, 2, SEEK_CUR);
        if (value < 0 || value > 9)
            return NULL;
        train_vectors_array[0][value] = 1.0;
        return train_vectors_array;
    }
    return NULL;
}
