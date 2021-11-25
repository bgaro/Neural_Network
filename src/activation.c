
#include "matrix.h"

void unit_step(matrix_t *m)
{
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            m->data[i][j] = (m->data[i][j] >= 0) ? 1 : 0;
        }
    }
}

float reLU(float x)
{
    return x >= 0 ? x : 0;
}

float reLU_derivative(float x)
{
    return x >= 0 ? 1 : 0;
}
