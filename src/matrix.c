#include <stdio.h>
#include <stdlib.h>
#include <math.h>
typedef struct
{
    int rows;
    int cols;
    float *data;
} matrix_t;

void matrix_copy(matrix_t *m, matrix_t *m_c)
{
    if (m->rows != m_c->rows || m->cols != m_c->cols)
    {
        printf("Error: matrix_copy: matrices have different dimensions\n");
        exit(1);
    }

    for (int i = 0; i < (m->rows * m->cols); i++)
    {
        m_c->data[i] = m->data[i];
    }
}

matrix_t *matrix_create(int rows, int cols)
{
    if (rows == 0 || cols == 0)
        return NULL;
    matrix_t *m = (matrix_t *)malloc(sizeof(matrix_t));
    m->rows = rows;
    m->cols = cols;
    m->data = (float *)calloc(sizeof(float), rows * cols);
    return m;
}

void matrix_print(matrix_t *m)
{
    if (m == NULL)
    {
        printf("Error matrix_print, matrix doesn't exists\n");
        return;
    }
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            printf("%f ", m->data[m->cols * i + j]);
        }
        printf("\n");
    }
}

void matrix_reset(matrix_t *m)
{
    if (m == NULL)
    {
        printf("Error matrix_reset, matrix doesn't exists\n");
        return;
    }
    for (int i = 0; i < m->rows * m->cols; i++)
    {
        m->data[i] = 0.0;
    }
}

void matrix_free(matrix_t *m)
{
    if (m == NULL)
    {
        printf("Error matrix_free, matrix doesn't exists\n");
        return;
    }
    free(m->data);
    free(m);
}

void matrix_add(matrix_t *m1, matrix_t *m2)
{
    if (m1 == NULL || m2 == NULL)
    {
        printf("Error matrix_add, matrix doesn't exists\n");
        return;
    }
    if ((m1->rows != m2->rows) || (m1->cols != m2->cols))
    {
        printf("Error: Matrix dimensions do not match\n");
        return;
    }
    for (int i = 0; i < m1->rows * m1->cols; i++)
    {

        m1->data[i] += m2->data[i];
    }
}

void matrix_transpose(matrix_t *m, matrix_t *m_transpose)
{
    if (m == NULL || m_transpose == NULL)
    {
        printf("Error matrix_transpose, matrix doesn't exists\n");
        return;
    }
    if ((m->rows != m_transpose->cols) || (m->cols != m_transpose->rows))
    {
        printf("Error: Matrix dimensions do not match\n");
        return;
    }
    matrix_reset(m_transpose);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            m_transpose->data[m_transpose->cols * j + i] = m->data[m->cols * i + j];
        }
    }
}

void matrix_initialize_random(matrix_t *m, int nb_neuron_out, int nb_neuron_in)

{
    if (m == NULL)
    {
        printf("Error matrix_initialize_random, matrix doesn't exists\n");
        return;
    }
    for (int i = 0; i < m->rows * m->cols; i++)
    {

        m->data[i] = (2 * (float)rand() / ((float)RAND_MAX) - 1) * sqrt(6.0) / sqrt(nb_neuron_out + nb_neuron_in);
    }
}

void matrix_initialize_to_value(matrix_t *m, float value)
{
    if (m == NULL)
    {
        printf("Error matrix_initialize_to_value, matrix doesn't exists\n");
        return;
    }
    for (int i = 0; i < m->rows * m->cols; i++)
    {

        m->data[i] = value;
    }
}

void matrix_diagonalize(matrix_t *m, matrix_t *m_diagonal)
{
    if (m == NULL || m_diagonal == NULL)
    {
        printf("Error matrix_diagonalize, matrix doesn't exists\n");
        return;
    }
    if ((m->rows != m_diagonal->rows) || (m->cols != 1))
    {
        printf("Error: Matrix dimensions do not match\n");
        return;
    }
    matrix_reset(m_diagonal);
    for (int i = 0; i < m->rows; i++)
    {
        m_diagonal->data[i * m_diagonal->cols + i] = m->data[i];
    }
}

void matrix_initialize(matrix_t *m, int rows, int cols, float **array)

{
    if (m == NULL || array == NULL)
    {
        printf("Error matrix_initize, matrix doesn't exists\n");
        return;
    }
    if (rows != m->rows || cols != m->cols)
    {
        printf("Error: Matrix dimensions do not match\n");
        return;
    }
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            m->data[m->cols * i + j] = array[i][j];
        }
    }
}

void matrix_multiply(matrix_t *m1, matrix_t *m2, matrix_t *m_mul)
{
    if (m1 == NULL || m2 == NULL || m_mul == NULL)
    {
        printf("Error matrix_multiply, matrix doesn't exists\n");
        return;
    }
    if ((m1->cols != m2->rows) || (m1->rows != m_mul->rows) || (m2->cols != m_mul->cols))
    {
        printf("Error: Matrix dimensions do not match\n");
        return;
    }
    matrix_reset(m_mul);
    for (int i = 0; i < m1->rows; i++)
    {
        for (int j = 0; j < m2->cols; j++)
        {
            for (int k = 0; k < m1->cols; k++)
            {

                m_mul->data[m_mul->cols * i + j] += m1->data[m1->cols * i + k] * m2->data[m2->cols * k + j];
            }
        }
    }
}

void matrix_multiply_constant(matrix_t *m, float c)
{
    if (m == NULL)
    {
        printf("Error matrix_multiply_constant, matrix doesn't exists\n");
        return;
    }
    for (int i = 0; i < m->rows * m->cols; i++)
    {

        m->data[i] *= c;
    }
}

void matrix_subtract(matrix_t *m1, matrix_t *m2)
{
    if (m1 == NULL || m2 == NULL)
    {
        printf("Error matrix_subtract, matrix doesn't exists\n");
        return;
    }
    if (m1->rows != m2->rows || m1->cols != m2->cols)
    {
        printf("Error: Matrix dimensions do not match\n");
        return;
    }
    for (int i = 0; i < m1->rows * m2->cols; i++)
    {

        m1->data[i] -= m2->data[i];
    }
}