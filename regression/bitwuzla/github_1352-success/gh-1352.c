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
    for(i=0; i<pow(2,n); i++)
    {
        for(j=0; j<pow(2,n); j++)
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
    __ESBMC_assume(n >= 0);
    __ESBMC_assume(n < 64 - 3); /* 3 = log2(sizeof(char *)) */
    char **fractal = (char **)malloc(pow(2,n) * sizeof(char *));
    int i, j;
    for(i=0; i<pow(2,n); i++)
        fractal[i] = (char *)malloc(pow(2,n) * sizeof(char));
    for(i=0; i<pow(2,n); i++)
        for(j=0; j<pow(2,n); j++)
            fractal[i][j] = ' ';
    generateFractal(fractal, n, 0, 0);
    printFractal(fractal, n);
    for(i=0; i<pow(2,n); i++)
        free(fractal[i]);
    free(fractal);
    return 0;
}
