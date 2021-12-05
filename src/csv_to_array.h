float **csv_to_array_vectors(FILE *train_vectors_stream, int size);

float **csv_to_array_labels(FILE *train_vectors_stream, int size);
int csv_to_array_labels_int(FILE *train_vectors_stream);
int get_label(matrix_t *labels);
