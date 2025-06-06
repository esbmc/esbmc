///FormAI DATASET v1.0 Category: Expense Tracker ; Style: peaceful
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define MAX_SIZE 100

struct expenses{
    char category[MAX_SIZE];
    float amount;
};

int main(){
    int n;
    float total_amount = 0;
    printf("Enter the number of expenses: ");
    scanf("%d", &n);
    struct expenses e[n]; // Creating an array of n expenses

    // Taking input from user
    for(int i=0; i<n; i++){
        printf("Enter category for expense %d: ", i+1);
        scanf("%s", e[i].category);
        printf("Enter amount for expense %d: ", i+1);
        scanf("%f", &e[i].amount);
        total_amount += e[i].amount;
    }

    // Printing table header
    printf("\nCATEGORY\tAMOUNT\n");
    printf("-----------------------\n");

    // Printing expenses and their amount
    for(int i=0; i<n; i++){
        printf("%s\t\t%.2f\n", e[i].category, e[i].amount);
    }

    // Printing total expenses
    printf("\nTotal Expenses: %.2f\n", total_amount);

    return 0;
}

