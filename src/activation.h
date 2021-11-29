
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
matrix_t *reLU(matrix_t *m);

/**
 * @brief ReLU_derivative activation functions
 * @param x input : inner potential
 * @return pointer to matrix 
 **/
matrix_t *reLU_derivate(matrix_t *m);

matrix_t *sigmoid(matrix_t *m);

matrix_t *sigmoid_derivative(matrix_t *m);