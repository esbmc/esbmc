//FormAI DATASET v1.0 Category: Matrix operations ; Style: funny
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

void party()
{
    printf("You have entered the party zone! Let's do some matrix operations\n");
}

void matrix_addition(int row, int col, int matrix1[row][col], int matrix2[row][col])
{
    int i,j;
    printf("\n\nMatrix Addition:\n");
    for(i=0;i<row;i++)
    {
        for(j=0;j<col;j++)
        {
            matrix1[i][j]+=matrix2[i][j];
            printf("%d\t",matrix1[i][j]);
        }
        printf("\n");
    }
}

void matrix_subtraction(int row, int col, int matrix1[row][col], int matrix2[row][col])
{
    int i,j;
    printf("\n\nMatrix Subtraction:\n");
    for(i=0;i<row;i++)
    {
        for(j=0;j<col;j++)
        {
            matrix1[i][j]-=matrix2[i][j];
            printf("%d\t",matrix1[i][j]);
        }
        printf("\n");
    }
}

void matrix_multiplication(int row1, int col1, int matrix1[row1][col1], int row2, int col2, int matrix2[row2][col2])
{
    int i,j,k;
    int result[row1][col2];
    printf("\n\nMatrix Multiplication:\n");
    for(i=0;i<row1;i++)
    {
        for(j=0;j<col2;j++)
        {
            result[i][j]=0;
            for(k=0;k<col1;k++)
            {
                result[i][j]+=matrix1[i][k]*matrix2[k][j];
            }
            printf("%d\t",result[i][j]);
        }
        printf("\n");
    }
}

void matrix_transpose(int row, int col, int matrix[row][col])
{
    int i,j;
    printf("\n\nMatrix Transpose:\n");
    for(i=0;i<col;i++)
    {
        for(j=0;j<row;j++)
        {
            printf("%d\t",matrix[j][i]);
        }
        printf("\n");
    }
}

int main()
{
    party();
    
    srand(time(0));
    
    int row1,col1,row2,col2,i,j,k;
    
    printf("Enter the number of rows for matrix 1: ");
    scanf("%d",&row1);
    
    printf("Enter the number of columns for matrix 1: ");
    scanf("%d",&col1);
    
    int matrix1[row1][col1];
    
    printf("\nMatrix 1:\n");
    for(i=0;i<row1;i++)
    {
        for(j=0;j<col1;j++)
        {
            matrix1[i][j]=rand()%10;
            printf("%d\t",matrix1[i][j]);
        }
        printf("\n");
    }
    
    printf("\nEnter the number of rows for matrix 2: ");
    scanf("%d",&row2);
    
    printf("Enter the number of columns for matrix 2: ");
    scanf("%d",&col2);
    
    int matrix2[row2][col2];
    
    printf("\nMatrix 2:\n");
    for(i=0;i<row2;i++)
    {
        for(j=0;j<col2;j++)
        {
            matrix2[i][j]=rand()%10;
            printf("%d\t",matrix2[i][j]);
        }
        printf("\n");
    }
    
    if(col1!=row2)
    {
        printf("\nMatrix multiplication not possible");
        exit(0);
    }
    
    matrix_addition(row1,col1,matrix1,matrix2);
    
    matrix_subtraction(row1,col1,matrix1,matrix2);
    
    matrix_multiplication(row1,col1,matrix1,row2,col2,matrix2);
    
    matrix_transpose(row1,col1,matrix1);
    
    matrix_transpose(row2,col2,matrix2);
    
    return 0;
}
