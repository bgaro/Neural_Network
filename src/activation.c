
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

        m_d->data[i] = (m->data[i] >= 0) ? m->data[i] : 0.01 * m->data[i];
    }
}

void softmax(matrix_t *m, matrix_t *m_d)
{
    if (m->rows != m_d->rows || m->cols != m_d->cols)
    {
        printf("Error softmax, matrix dimensions don't match\n");
        return;
    }
    double sum = 0;
    for (int i = 0; i < m->rows * m->cols; i++)
    {
        sum += exp(m->data[i]);
    }
    for (int i = 0; i < m->rows * m->cols; i++)
    {
        m_d->data[i] = exp(m->data[i]) / sum;
    }
}

// compute ther derivate of softmax function
void softmax_derivate(matrix_t *m, matrix_t *m_d)
{
    if (m->rows != m_d->rows || m->cols != m_d->cols)
    {
        printf("Error softmax_derivative, matrix dimensions don't match\n");
        return;
    }
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            if (i == j)
                m_d->data[i * m->cols + j] = m->data[i * m->cols + j] * (1 - m->data[i * m->cols + j]);
            else
                m_d->data[i * m->cols + j] = -m->data[i * m->cols + j] * m->data[i * m->cols + j];
        }
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

        m_d->data[i] = (m->data[i] >= 0) ? 1 : 0.01;
    }
}
