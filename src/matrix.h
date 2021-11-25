typedef struct
{
    int rows;
    int cols;
    float **data;
} matrix_t;

/**
 * @brief Creates a new matrix with the given dimensions.
 * @param rows The number of rows.
 * @param cols The number of columns.
 * @return A pointer to the new matrix.
 */
matrix_t *matrix_create(int rows, int cols);

/**
 * @brief multiply given matrix by constant c
 * @param m The matrix to multiply.
 * @param c The constant to multiply by.
 * @return void
 **/
void matrix_multiply_constant(matrix_t *m, float c);

/**
 * @brief multiply matrix m1 to matrix m2
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @return pointer to new matrix.
 * */
matrix_t *matrix_multiply(matrix_t *m1, matrix_t *m2);

/**
 * @brief add matrix m1 to matrix m2
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @return pointer to new matrix.
 * */
matrix_t *matrix_add(matrix_t *m1, matrix_t *m2);

/**
 * @brief free the given matrix
 * @param m The matrix to free.
 * @return void
 * */
void matrix_free(matrix_t *m);

/**
 * @brief print the given matrix
 * @param m The matrix to print.
 * @return void
 * */
void matrix_print(matrix_t *m);

/**
 * @brief transpose the given matrix
 * @param m The matrix to transpose.
 * @return pointer to new matrix.
 * */
matrix_t *matrix_transpose(matrix_t *m);

/**
 * @brief subtract matrix m2 from matrix m1
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @return pointer to new matrix.
 * */
matrix_t *matrix_subtract(matrix_t *m1, matrix_t *m2);
