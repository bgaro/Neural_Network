typedef struct
{
    int rows;
    int cols;
    float *data;
} matrix_t;

/**
 * @brief Creates a new matrix with the given dimensions.
 * @param rows The number of rows.
 * @param cols The number of columns.
 * @return A pointer to the new matrix.
 */
matrix_t *matrix_create(int rows, int cols);

/**
 * @brief copy the given matrix
 * @param m The matrix to copy.
 * @param m_copy The matrix to copy to.
 * @return void
 * */
void matrix_copy(matrix_t *m, matrix_t *m_copy);

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
 * @param m_mul The matrix to store the result.
 * @return void
 * */
void matrix_multiply(matrix_t *m1, matrix_t *m2, matrix_t *m_mul);

/**
 * @brief add matrix m1 to matrix m2
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * */
void matrix_add(matrix_t *m1, matrix_t *m2);

void matrix_hadamard(matrix_t *m_1, matrix_t *m_2, matrix_t *m_h);
/**
 * @brief reset matrix m to 0
 * @param m The first matrix.
 * @return void
 * */
void matrix_reset(matrix_t *m);

/**
 * @brief free the given matrix
 * @param m The matrix to free.
 * @return void
 * */
void matrix_free(matrix_t *m);

/**
 * @brief Diagonalize a vector
 * @param m The vector to diagonalize. (col = 1)
 * @param m_diagonal The matrix to store the diagonalized vector.
 * @return void
 * */
void matrix_diagonalize(matrix_t *m, matrix_t *m_diagonal);

/**
 * @brief initialize the given matrix with random values
 * @param m The matrix to initialize.
 * @return void
 * */
void matrix_initialize_random(matrix_t *m, int nb_neuron_out, int nb_neuron_in);

void matrix_initialize_to_value(matrix_t *m, float value);
/**
 * @brief transpose the given matrix
 * @param m The matrix to transpose.
 * @param m_transpose The matrix to store the transposed matrix.
 * @return void
 * */
void matrix_transpose(matrix_t *m, matrix_t *m_transpose);
/**
 * @brief print the given matrix
 * @param m The matrix to print.
 * @return void
 * */
void matrix_print(matrix_t *m);

/**
 * @brief subtract matrix m2 from matrix m1
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * */
void matrix_subtract(matrix_t *m1, matrix_t *m2);

/**
 * @brief initialize matrix data with array values
 * @param m The matrix to initialize.
 * @param rows The number of rows.
 * @param cols The number of columns.
 * @param array The array of values.
 * @return void
 * */
void matrix_initialize(matrix_t *m, int rows, int cols, float *array);
