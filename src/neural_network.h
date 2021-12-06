#define SOFTMAX 0
#define RELU 1

/**
 * @brief Feed forward function compute the output of a layer given the input and the weight matrix
 * @param weights weight matrix between the input layer and the actual layer
 * @param input The input matrix (xi)
 * @param bias bias matrix
 * @param output output matrix that will contained the inner potential of the current layer
 * @param activation_output activation matrix that will contained the activation function of the inner potential of the current layer
 * @param activation activation function of the current layer
 * @return void
 */
void feed_forward(matrix_t *weights, matrix_t *input, matrix_t *bias, matrix_t *output, matrix_t *activation_output, int activation);

/**
 * @brief Backpropagation function compute the gradient of the cost function with respect to the weight matrix
 * @param derivate_error derivate of the cost function with respect to the output of the upper layer
 * @param derivate_activation derivate of the activation function of the output of the upper layer
 * @param derivate_error_activation matrix that will contains the multiplication of the derivate_error and the derivate_activation
 * @param derivate_error_activation_transpose matrix that will contains the transpose of the derivate_error_activation matrix
 * @param weights weight matrix between the current layer and the upper layer
 * @param derivate_error_output_transpose transposed matrix of the derivate_error_output matrix
 * @param derivate_error_output matrix that will contains the result of the function
 * @param activation activation function of the previous layer
 * @return void
 */
void backward_propagation_neurons(matrix_t *derivate_error, matrix_t *derivate_activation, matrix_t *derivate_error_activation, matrix_t *derivate_error_activation_transpose, matrix_t *weights, matrix_t *derivate_error_output_transpose, matrix_t *derivate_error_output, int activation);

/**
 * @brief Backpropagation function compute the gradient of the cost function with respect to the weight matrix
 * @param derivate_error derivate of the cost function with respect to the output of the upper layer
 * @param derivate_activation derivate of the activation function of the output of the upper layer
 * @param derivate_error_activation matrix that will contains the multiplication of the derivate_error and the derivate_activation
 * @param activation_layer activation function of the lower layer
 * @param activation_layer_transpose transposed matrix of the activation_layer matrix
 * @param weight_derivate_output matrix that will contains the result of the function
 * @param activation activation function of the previous layer
 * @return void
 * */
void backward_propagation_weights(matrix_t *derivate_error, matrix_t *derivate_activation, matrix_t *derivate_error_activation, matrix_t *activation_layer, matrix_t *activation_layer_transpose, matrix_t *weight_derivate_output, int activation);
