#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void matrix_print(int ROW, int COL, double A[ROW][COL]);

void matrix_vector_multiplication(double * input_vector, int INPUT_LEN, double * output_vector,
		int OUTPUT_LEN, double weights_matrix[OUTPUT_LEN][INPUT_LEN]);


/**
 * @brief Multiply input_matrix1 by input_matrix2 and story it in output_matrix
 * @param input_matrix1 The first matrix.
 * @param input_matrix2 The second matrix.
 * @param output_matrix The matrix to store the result.
 * @return void
 * */
void matrix_matrix_multiplication(uint32_t MATRIX1_ROW, uint32_t MATRIX1_COL, uint32_t MATRIX2_COL,
									double input_matrix1[MATRIX1_ROW][MATRIX1_COL],
									double input_matrix2[MATRIX1_COL][MATRIX2_COL],
									double output_matrix[MATRIX1_ROW][MATRIX2_COL]);

void matrix_matrix_sub(uint32_t MATRIX_ROW, uint32_t MATRIX_COL,
									double input_matrix1[MATRIX_ROW][MATRIX_COL],
									double input_matrix2[MATRIX_ROW][MATRIX_COL],
									double output_matrix[MATRIX_ROW][MATRIX_COL]);

void matrix_transpose(uint32_t ROW, uint32_t COL, double A[ROW][COL], double A_T[COL][ROW]);
