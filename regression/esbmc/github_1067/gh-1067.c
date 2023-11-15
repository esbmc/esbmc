//FormAI DATASET v1.0. Category: Memory management ; Style: calm
#include<stdio.h>
#include<stdlib.h>

int main(){
    int* ptr;
    int size;
    printf("Enter number of integers: ");
    scanf("%d", &size);

    // Allocating memory dynamically for array
    ptr = (int*) malloc(size * sizeof(int));
    if(ptr == NULL){
        printf("Memory allocation failed.\n");
        exit(0);
    }

    // Initializing the array
    for(int i = 0; i < size; i++){
        *(ptr + i) = i + 1;
    }

    // Displaying the array
    printf("The array is: ");
    for(int i = 0; i < size; i++){
        printf("%d ", *(ptr + i));
    }

    // Reallocating memory for array
    printf("\nEnter new size of array: ");
    scanf("%d", &size);
    ptr = realloc(ptr, size * sizeof(int));
    if(ptr == NULL){
        printf("Memory reallocation failed.\n");
        exit(0);
    }

    // Adding new elements to the array
    printf("Enter %d new elements of the array:\n", size);
    for(int i = size - 5; i < size; i++){
        scanf("%d", &*(ptr + i));
    }

    // Displaying the new array
    printf("The new array is: ");
    for(int i = 0; i < size; i++){
        printf("%d ", *(ptr + i));
    }

    // Freeing the allocated memory
    free(ptr);
    printf("\nMemory freed successfully.\n");
    return 0;
}
