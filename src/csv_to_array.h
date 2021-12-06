/**
 * @brief Get an array of all values of each vector from the train vector stream (retrieved from csv file)
 * @param train_vectors_stream The stream obtained from the csv file
 * @param size Size of the stream, i.e the number of rows in the csv file in our case
 * @return An array of float, with values of vectors from the csv file
 **/
float **csv_to_array_vectors(FILE *train_vectors_stream, int size);

/**
 * @brief Get an array of each label from the train vector stream (retrieved from csv file)
 * @param train_vectors_stream The stream obtained from the csv file
 * @param size Size of the stream, i.e the number of rows in the csv file in our case
 * @return An array of float, with values of labels from the csv file
 **/
float **csv_to_array_labels(FILE *train_vectors_stream, int size);

/**
 * @brief Get the label of each vector as a integer format
 * @param train_vectors_stream The stream obtained from the csv file
 * @return The label as an integer
 **/
int csv_to_array_labels_int(FILE *train_vectors_stream);

/**
 * @brief Get the label from the output activation matrix, will be compared to the label extracted from the csv file
 * @param labels The output activation matrix
 * @return The label as an integer
 **/
int get_label(matrix_t *labels);