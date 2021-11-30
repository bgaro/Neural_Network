
/**
 * @brief unit step activation functions
 * @param m input : matrix to apply activation function to
 **/
matrix_t *unit_step(matrix_t *m);
/**
 * @brief ReLU activation functions
 * @param x input : inner potential
 * @return pointer to matrix
 **/
void reLU(matrix_t *m, matrix_t *m_d);

/**
 * @brief ReLU_derivative activation functions
 * @param x input : inner potential
 * @return pointer to matrix
 **/
void reLU_derivate(matrix_t *m, matrix_t *m_d);

void sigmoid(matrix_t *m, matrix_t *m_d);

void sigmoid_derivate(matrix_t *m, matrix_t *m_d);