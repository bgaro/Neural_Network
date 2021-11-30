
#include "matrix.h"
#include <math.h>
#include <stdio.h>

void unit_step(matrix_t *m, matrix_t *m_d)
{
    if (m->rows != m_d->rows || m->cols != m_d->cols)
    {
        printf("Error sigmoid_derivate, matrix dimensions don't match\n");
        return;
    }
    for (int i = 0; i < m->rows * m->cols; i++)
    {

        m_d->data[i] = (m->data[i] >= 0) ? 1 : 0;
    }
}
void sigmoid(matrix_t *m, matrix_t *m_d)
{
    if (m->rows != m_d->rows || m->cols != m_d->cols)
    {
        printf("Error sigmoid_derivate, matrix dimensions don't match\n");
        return;
    }
    for (int i = 0; i < m->rows * m->cols; i++)
    {

        m_d->data[i] = 1 / (1 + exp(-m->data[i]));
    }
}

void sigmoid_derivate(matrix_t *m, matrix_t *m_d)
{
    if (m->rows != m_d->rows || m->cols != m_d->cols)
    {
        printf("Error sigmoid_derivate, matrix dimensions don't match\n");
        return;
    }
    for (int i = 0; i < m->rows * m->cols; i++)
    {

        m_d->data[i] = m->data[i] * (1 - m->data[i]);
    }
}
void reLU(matrix_t *m, matrix_t *m_d)
{

    if (m->rows != m_d->rows || m->cols != m_d->cols)
    {
        printf("Error sigmoid_derivate, matrix dimensions don't match\n");
        return;
    }
    for (int i = 0; i < m->rows * m->cols; i++)
    {

        m_d->data[i] = (m->data[i] >= 0) ? m->data[i] : 0;
    }
}

void reLU_derivate(matrix_t *m, matrix_t *m_d)
{

    if (m->rows != m_d->rows || m->cols != m_d->cols)
    {
        printf("Error sigmoid_derivate, matrix dimensions don't match\n");
        return;
    }
    for (int i = 0; i < m->rows * m->cols; i++)
    {

        m_d->data[i] = (m->data[i] >= 0) ? 1 : 0;
    }
}
