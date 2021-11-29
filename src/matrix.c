#include <stdio.h>
#include <stdlib.h>
typedef struct
{
    int rows;
    int cols;
    float **data;
} matrix_t;

matrix_t *matrix_copy(matrix_t *m)
{
    matrix_t *copy = malloc(sizeof(matrix_t));
    copy->rows = m->rows;
    copy->cols = m->cols;
    copy->data = malloc(sizeof(float *) * m->rows);
    for (int i = 0; i < m->rows; i++)
    {
        copy->data[i] = malloc(sizeof(float) * m->cols);
        for (int j = 0; j < m->cols; j++)
        {
            copy->data[i][j] = m->data[i][j];
        }
    }
    return copy;
}

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

matrix_t *matrix_sum(matrix_t *m)
{
    matrix_t *result = matrix_create(1, m->cols);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            result->data[0][j] += m->data[i][j];
        }
    }
    return result;
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
    for (int i = 0; i < m1->rows; i++)
    {
        for (int j = 0; j < m1->cols; j++)
        {
            m1->data[i][j] = m1->data[i][j] + m2->data[i][j];
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
    matrix_t *transpose = matrix_create(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            transpose->data[j][i] = m->data[i][j];
        }
    }
    return transpose;
}
void matrix_initialize_random(matrix_t *m, int seed)
{
    if (m == NULL)
    {
        printf("Error matrix_initialize_random, matrix doesn't exists\n");
        return;
    }
    srand(seed);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            m->data[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }
}
matrix_t *matrix_diagonalize(matrix_t *m)
{
    if (m == NULL)
    {
        printf("Error matrix_diagonalize, matrix doesn't exists\n");
        return NULL;
    }
    if (m->cols != 1)
    {
        printf("Error: Matrix is not a vector\n");
        return NULL;
    }
    matrix_t *diag = matrix_create(m->rows, m->rows);
    for (int i = 0; i < m->rows; i++)
    {
        diag->data[i][i] = m->data[i][0];
    }
    return diag;
}

void matrix_initialize(matrix_t *m, int rows, int cols, float array[rows][cols])

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
            m->data[i][j] = array[i][j];
        }
    }
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
    for (int i = 0; i < m1->rows; i++)
    {
        for (int j = 0; j < m1->cols; j++)
        {
            m1->data[i][j] = m1->data[i][j] - m2->data[i][j];
        }
    }
}