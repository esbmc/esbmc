//FormAI DATASET v1.0 Category: Fractal Generation ; Style: future-proof
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* function to get the user input */
int getUserInput()
{
    int n;
    printf("Enter the depth of the fractal: ");
    scanf("%d", &n);
    return n;
}

/* function to print fractal */
void printFractal(char **fractal, int n)
{
    int i, j;
    double N = pow(2,n);
    for(i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
            printf("%c", fractal[i][j]);
        }
        printf("\n");
    }
}

/* function to generate fractal */
void generateFractal(char **fractal, int n, int x, int y)
{
    int i, j;
    if(n==0)
        fractal[x][y] = '*';
    else
    {
        int length = pow(2,n-1);
        generateFractal(fractal, n-1, x, y);
        generateFractal(fractal, n-1, x+length, y);
        generateFractal(fractal, n-1, x, y+length);
        generateFractal(fractal, n-1, x+length, y+length);
    }
}

int main()
{
    int n = getUserInput();
    double N = pow(2,n);
    char **fractal = (char **)malloc(N * sizeof(char *));
    int i, j;
    for(i=0; i<N; i++)
        fractal[i] = (char *)malloc(N * sizeof(char));
    for(i=0; i<N; i++)
        for(j=0; j<N; j++)
            fractal[i][j] = ' ';
    generateFractal(fractal, n, 0, 0);
    printFractal(fractal, n);
    for(i=0; i<N; i++)
        free(fractal[i]);
    free(fractal);
    return 0;
}
