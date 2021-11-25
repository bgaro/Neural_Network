
/**
 * @brief unit step activation functions
 * @param m input : matrix to apply activation function to
 **/
void unit_step(matrix_t *m);

/**
 * @brief ReLU activation functions
 * @param x input : inner potential
 * @return 0 if x<0, x if x>=0
 **/
float reLU(float x);

/**
 * @brief ReLU_derivative activation functions
 * @param x input : inner potential
 * @return 0 if x<0, 1 if x>=0
 **/
float reLU_derivative(float x);