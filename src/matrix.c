#include <stdio.h>

typedef struct
{
    int rows;
    int cols;
    float **data;
} matrix_t;

matrix_t *matrix_create(int rows, int cols)
{
    if (rows == 0 || cols == 0)
        return NULL;
    matrix_t *m = (matrix_t *)malloc(sizeof(matrix_t));
    m->rows = rows;
    m->cols = cols;
    m->data = (float **)malloc(sizeof(float *) * rows);
    for (int i = 0; i < rows; i++)
    {
        m->data[i] = (float *)malloc(sizeof(float) * cols);
    }
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
            printf("%f ", m->data[i][j]);
        }
        printf("\n");
    }
}

void matrix_free(matrix_t *m)
{
    if (m == NULL)
    {
        printf("Error matrix_free, matrix doesn't exists\n");
        return;
    }
    for (int i = 0; i < m->rows; i++)
    {
        free(m->data[i]);
    }
    free(m->data);
    free(m);
}

matrix_t *matrix_add(matrix_t *m1, matrix_t *m2)
{
    if (m1 == NULL || m2 == NULL)
    {
        printf("Error matrix_add, matrix doesn't exists\n");
        return NULL;
    }
    if (m1->rows != m2->rows || m1->cols != m2->cols)
    {
        printf("Error: Matrix dimensions do not match\n");
        return NULL;
    }
    matrix_t *m = matrix_create(m1->rows, m1->cols);
    for (int i = 0; i < m1->rows; i++)
    {
        for (int j = 0; j < m1->cols; j++)
        {
            m->data[i][j] = m1->data[i][j] + m2->data[i][j];
        }
    }
    return m;
}

matrix_t *matrix_multiply(matrix_t *m1, matrix_t *m2)
{
    if (m1 == NULL || m2 == NULL)
    {
        printf("Error matrix_multiply, matrix doesn't exists\n");
        return NULL;
    }
    if (m1->cols != m2->rows)
    {
        printf("Error: Matrix dimensions do not match\n");
        return NULL;
    }
    matrix_t *m = matrix_create(m1->rows, m2->cols);
    for (int i = 0; i < m1->rows; i++)
    {
        for (int j = 0; j < m2->cols; j++)
        {
            float sum = 0;
            for (int k = 0; k < m1->cols; k++)
            {
                sum += m1->data[i][k] * m2->data[k][j];
            }
            m->data[i][j] = sum;
        }
    }
    return m;
}

void matrix_multiply_constant(matrix_t *m, float c)
{
    if (m == NULL)
    {
        printf("Error matrix_multiply_constant, matrix doesn't exists\n");
        return;
    }
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            m->data[i][j] *= c;
        }
    }
}

matrix_t *matrix_transpose(matrix_t *m)
{
    if (m == NULL)
    {
        printf("Error matrix_transpose, matrix doesn't exists\n");
        return NULL;
    }
    matrix_t *m_t = matrix_create(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            m_t->data[j][i] = m->data[i][j];
        }
    }
    return m_t;
}

matrix_t *matrix_subtract(matrix_t *m1, matrix_t *m2)
{
    if (m1 == NULL || m2 == NULL)
    {
        printf("Error matrix_subtract, matrix doesn't exists\n");
        return NULL;
    }
    if (m1->rows != m2->rows || m1->cols != m2->cols)
    {
        printf("Error: Matrix dimensions do not match\n");
        return NULL;
    }
    matrix_t *m = matrix_create(m1->rows, m1->cols);
    for (int i = 0; i < m1->rows; i++)
    {
        for (int j = 0; j < m1->cols; j++)
        {
            m->data[i][j] = m1->data[i][j] - m2->data[i][j];
        }
    }
    return m;
}