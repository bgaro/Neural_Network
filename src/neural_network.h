#define SOFTMAX 0
#define RELU 1

typedef struct
{
    matrix_t *a;
    matrix_t *b;
    matrix_t *c;
    int *index;
} args;

void feed_forward(matrix_t *weights, matrix_t *input, matrix_t *bias, matrix_t *output, matrix_t *activation_output, void (*activation_function)(matrix_t *, matrix_t *), int max_thread, pthread_t thread[max_thread], args *argument[max_thread]);
void backward_propagation_neurons(matrix_t *derivate_error, matrix_t *derivate_activation, matrix_t *derivate_error_activation, matrix_t *derivate_error_activation_transpose, matrix_t *weights, matrix_t *derivate_error_output_transpose, matrix_t *derivate_error_output, int activation, int max_thread, pthread_t thread[max_thread], args *argument[max_thread]);

void backward_propagation_weights(matrix_t *derivate_error, matrix_t *derivate_activation, matrix_t *derivate_error_activation, matrix_t *activation_layer, matrix_t *activation_layer_transpose, matrix_t *weight_derivate_output, int activation, int max_thread, pthread_t thread[max_thread], args *argument[max_thread]);
