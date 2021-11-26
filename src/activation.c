
#include "matrix.h"

matrix_t *unit_step(matrix_t *m)
{
    matrix_t *result = matrix_create(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            result->data[i][j] = (m->data[i][j] >= 0) ? 1 : 0;
        }
    }
    return result;
}

matrix_t *reLU(matrix_t *m)
{
    matrix_t *result = matrix_create(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            result->data[i][j] = (m->data[i][j] >= 0) ? 1 : 0;
        }
    }
    return result;
}

matrix_t *reLU_derivate(matrix_t *m)
{
    matrix_t *result = matrix_create(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            result->data[i][j] = (m->data[i][j] >= 0) ? m->data[i][j] : 0;
        }
    }
    return result;
}
