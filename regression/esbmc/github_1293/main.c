
//FormAI DATASET  v1.1 Category: Online Examination System ; Style: brave
#include <stdio.h>
#include <stdlib.h>

int main(){
    printf("Welcome to the Online Examination System\n");
    int score = 0;
    char answer[20];

    printf("Question 1: What is the output of 'printf(\"Hello World!\\n\");'? \n");
    printf("A. Hello World!\n");
    printf("B. HelloWorld!\n");
    printf("C. hello world!\n");
    printf("D. Hello_World!\n");
    scanf("%s", answer);
    if(answer[0] == 'A' || answer[0] == 'a'){
        score += 10;
    }

    printf("Question 2: How do you declare an integer variable in C? \n");
    printf("A. int num;\n");
    printf("B. float num;\n");
    printf("C. char num;\n");
    printf("D. num is not a variable type in C\n");
    scanf("%s", answer);
    if(answer[0] == 'A' || answer[0] == 'a'){
        score += 10;
    }

    printf("Question 3: What is the output of '2 + 2 * 3 - 1'?\n");
    printf("A. 6\n");
    printf("B. 7\n");
    printf("C. 8\n");
    printf("D. 9\n");
    scanf("%s", answer);
    if(answer[0] == 'B' || answer[0] == 'b'){
        score += 10;
    }

    printf("Question 4: What is the purpose of the 'return' statement in a function? \n");
    printf("A. To print a value\n");
    printf("B. To exit the program\n");
    printf("C. To stop the current function and return a value\n");
    printf("D. To start a new function\n");
    scanf("%s", answer);
    if(answer[0] == 'C' || answer[0] == 'c'){
        score += 10;
    }

    printf("Question 5: What is the keyword used to declare a function in C? \n");
    printf("A. int\n");
    printf("B. void\n");
    printf("C. function\n");
    printf("D. declare\n");
    scanf("%s", answer);
    if(answer[0] == 'B' || answer[0] == 'b'){
        score += 10;
    }

    printf("\nYour Score: %d/50\n", score);

    if(score >= 40){
        printf("Congratulations! You have passed the exam!\n");
    } else {
        printf("Sorry, you have not passed the exam. Please try again.\n");
    }

    return 0;
}
