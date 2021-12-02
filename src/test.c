#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main()
{
    matrix_t *matrix = matrix_create(3, 1);
    matrix_t *matrix2 = matrix_create(1, 3);
    matrix_t *matrix3 = matrix_create(3, 3);
    float **data = malloc(sizeof(float *) * 3);
    for (int i = 0; i < 3; i++)
    {
        data[i] = malloc(sizeof(float) * 1);
        for (int j = 0; j < 2; j++)
        {
            data[i][j] = i + j;
        }
    }
    matrix_initialize(matrix, 3, 1, data);
    matrix_transpose(matrix, matrix2);
    matrix_multiply(matrix, matrix2, matrix3);
    matrix_print(matrix2);
    printf("\n");
    matrix_print(matrix);
    printf("\n");

    matrix_multiply_constant(matrix3, 2);
    matrix_subtract(matrix3, matrix3);
    matrix_print(matrix3);
    return 0;
}