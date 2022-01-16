

/**
 * @brief unit step activation functions
 * @param m input : inner potential
 * @param m_d result of the activation function
 * @return void
 **/
void unit_step(matrix_t *m, matrix_t *m_d);

/**
 * @brief ReLU activation functions
 * @param m input : inner potential
 * @param m_d result of the activation function
 * @return void
 **/
void reLU(matrix_t *m, matrix_t *m_d);

/**
 * @brief derivate of ReLU activation functions
 * @param m input : inner potential
 * @param m_d result of the activation function
 * @return void
 **/
void reLU_derivate(matrix_t *m, matrix_t *m_d);

/**
 * @brief sigmoid activation functions
 * @param m input : inner potential
 * @param m_d result of the activation function
 * @return void
 **/
void sigmoid(matrix_t *m, matrix_t *m_d);

/**
 * @brief derivate of sigmoid activation functions
 * @param m input : inner potential
 * @param m_d result of the activation function
 * @return void
 **/
void sigmoid_derivate(matrix_t *m, matrix_t *m_d);

/**
 * @brief softmax activation functions
 * @param m input : inner potential
 * @param m_d result of the activation function
 * @return void
 **/
void softmax(matrix_t *m, matrix_t *m_d);

/**
 * @brief derivate of softmax activation functions
 * @param m input : inner potential
 * @param m_d result of the activation function
 * @return void
 **/
void softmax_derivate(matrix_t *m, matrix_t *m_d);

void elu(matrix_t *m, matrix_t *m_d);

void elu_derivate(matrix_t *m, matrix_t *m_d);
