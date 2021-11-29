#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TRAIN_VECTORS_LENGTH 2968 // Tests have been made
#define IMAGE_SIZE 784 // Images 28*28

int main() {
    int column = 0, cpt = 0;
    char * train_vectors;
    int train_vectors_array[1][IMAGE_SIZE];
    FILE * train_vectors_stream;
    char * train_vectors_string;
    train_vectors_string = malloc(sizeof(char) * MAX_TRAIN_VECTORS_LENGTH);

    char * file = "../data/fashion_mnist_train_vectors.csv";
    //char * file = "../data/test.csv";

    if((train_vectors_stream = fopen(file, "r")) == NULL) {
        perror("Error opening the CSV file");
        exit(EXIT_FAILURE);
    }

    while((fgets(train_vectors_string, MAX_TRAIN_VECTORS_LENGTH, train_vectors_stream)) != NULL) {
        
        train_vectors = strtok(train_vectors_string, ",");

        while(train_vectors != NULL) {
            train_vectors_array[0][column] = atoi(train_vectors);
            train_vectors = strtok(NULL, ",");
            column += 1;
        }
        
        memset(train_vectors_string, 0, MAX_TRAIN_VECTORS_LENGTH);
        column = 0;

        /* AFFICHAGE CLEAN DES IMAGES */
        printf("***** IMAGE %i *****\n\n", cpt);
        for(int i = 0; i < IMAGE_SIZE; i++) {
            if(i%28 == 0 && i != 0)
                printf("\n");
            printf("%-4i ", train_vectors_array[0][i]);
        }
        printf("\n\n\n\n\n");
        cpt += 1;
    }
}