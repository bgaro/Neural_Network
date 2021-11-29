#include <string.h>

#define MAX_TRAIN_VECTORS_LENGTH 2968 // Tests have been made
#define IMAGE_SIZE 784 // Images 28*28


float ** csv_to_array_vectors(char * file, FILE * train_vectors_stream) {

    /* Initializations */
    int column = 0, cpt = 1;
    char * train_vectors;
    
    char * train_vectors_string;
    train_vectors_string = malloc(sizeof(char) * MAX_TRAIN_VECTORS_LENGTH);
    float ** train_vectors_array;
    train_vectors_array = malloc(IMAGE_SIZE*sizeof(float));
    for(int i = 0; i < IMAGE_SIZE; i++) {
        train_vectors_array[i] = malloc(sizeof(float));
    }


    /* Reading the whole csv line by line */
    while((fgets(train_vectors_string, MAX_TRAIN_VECTORS_LENGTH, train_vectors_stream)) != NULL) {
        
        train_vectors = strtok(train_vectors_string, ",");

        /* Adding properly the vector to the array of all vectors */
        while(train_vectors != NULL) {
            train_vectors_array[0][column] = atoi(train_vectors);
            train_vectors = strtok(NULL, ",");
            column += 1;
        }
        
        /* Reset values to read next ones */
        memset(train_vectors_string, 0, MAX_TRAIN_VECTORS_LENGTH);
        column = 0;

        /* Print pictures */
        /*printf("***** IMAGE %i *****\n\n", cpt);
        for(int i = 0; i < IMAGE_SIZE; i++) {
            if(i%28 == 0 && i != 0)
                printf("\n");
            printf("%-4.0f", train_vectors_array[0][i]);
        }
        printf("\n\n\n\n\n");
        cpt += 1;*/
        fseek(train_vectors_stream, 1, SEEK_CUR);
        return train_vectors_array;
    }
}

void csv_to_array_labels(char * csv_file) {

    /* Initializations */
    int value;
    FILE * train_vectors_stream;
    char * train_vectors_string;

    /* Test on the opening of the given file */
    if((train_vectors_stream = fopen(csv_file, "r")) == NULL) {
        perror("Error opening the CSV file");
        exit(EXIT_FAILURE);
    }

    /* Getting the class of each vector */
    while((value = fgetc(train_vectors_stream)) != EOF) {
        value = value - 48;
        printf("%i\n", value);

        fseek(train_vectors_stream, 2, SEEK_CUR);
    }
}
