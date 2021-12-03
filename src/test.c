#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "activation.h"

int main()
{
    matrix_t *matrix = matrix_create(3, 1);
    matrix_t *matrix2 = matrix_create(1, 3);
    matrix_t *matrix3 = matrix_create(3, 3);
    float **data = malloc(sizeof(float *) * 3);
    for (int i = 0; i < 3; i++)
    {
        data[i] = malloc(sizeof(float));
        for (int j = 0; j < 1; j++)
        {
            data[i][j] = i + j + 1;
        }
    }

    matrix_initialize(matrix, 3, 1, data);
    printf("test\n");
    softmax_derivate(matrix, matrix3);
    matrix_print(matrix3);

    return 0;
}