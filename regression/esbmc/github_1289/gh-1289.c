//FormAI DATASET v1.0 Category: Matrix operations ; Style: complex
#include <stdio.h>

#define MATRIX_SIZE 3

typedef struct {
    float real;
    float imag;
} Complex;

void print_complex(Complex num) {
    printf("%f", num.real);
    if (num.imag >= 0) {
        printf("+%fi", num.imag);
    } else {
        printf("%fi", num.imag);
    }
}

void add_matrices(Complex matrix1[][MATRIX_SIZE], Complex matrix2[][MATRIX_SIZE], Complex result[][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            result[i][j].real = matrix1[i][j].real + matrix2[i][j].real;
            result[i][j].imag = matrix1[i][j].imag + matrix2[i][j].imag;
        }
    }
}

void multiply_matrices(Complex matrix1[][MATRIX_SIZE], Complex matrix2[][MATRIX_SIZE], Complex result[][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            result[i][j].real = 0;
            result[i][j].imag = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                Complex product;
                product.real = matrix1[i][k].real * matrix2[k][j].real - matrix1[i][k].imag * matrix2[k][j].imag;
                product.imag = matrix1[i][k].real * matrix2[k][j].imag + matrix1[i][k].imag * matrix2[k][j].real;

                result[i][j].real += product.real;
                result[i][j].imag += product.imag;
            }
        }
    }
}

int main() {
    Complex matrix1[MATRIX_SIZE][MATRIX_SIZE] = {
        { {1, 2}, {3, 4}, {5, 6} },
        { {7, 8}, {9, 10}, {11, 12} },
        { {13, 14}, {15, 16}, {17, 18} }
    };

    Complex matrix2[MATRIX_SIZE][MATRIX_SIZE] = {
        { {2, 1}, {4, 3}, {6, 5} },
        { {8, 7}, {10, 9}, {12, 11} },
        { {14, 13}, {16, 15}, {18, 17} }
    };

    Complex matrix_sum[MATRIX_SIZE][MATRIX_SIZE];
    Complex matrix_product[MATRIX_SIZE][MATRIX_SIZE];

    add_matrices(matrix1, matrix2, matrix_sum);
    multiply_matrices(matrix1, matrix2, matrix_product);

    printf("Matrix 1:\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            print_complex(matrix1[i][j]);
            printf("\t");
        }
        printf("\n");
    }
    printf("\n");

    printf("Matrix 2:\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            print_complex(matrix2[i][j]);
            printf("\t");
        }
        printf("\n");
    }
    printf("\n");

    printf("Matrix Sum:\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            print_complex(matrix_sum[i][j]);
            printf("\t");
        }
        printf("\n");
    }
    printf("\n");

    printf("Matrix Product:\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            print_complex(matrix_product[i][j]);
            printf("\t");
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}
