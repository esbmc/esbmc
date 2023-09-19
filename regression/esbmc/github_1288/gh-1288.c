//FormAI DATASET v1.0 Category: Online Examination System ; Style: careful
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define maximum number of questions and options
#define MAX_QUESTIONS 10
#define MAX_OPTIONS 4

// Struct for each question and its options
typedef struct {
    char question[100];
    char options[MAX_OPTIONS][50];
    int answer;
} Question;

// Struct for the exam
typedef struct {
    char title[50];
    int totalQuestions;
    Question questions[MAX_QUESTIONS];
    int correctAnswers;
} Exam;

// Function to display the exam questions and choices
void displayExam(Exam exam) {
    printf("**********\n");
    printf("%s\n\n", exam.title);
    for(int i=0; i<exam.totalQuestions; i++) {
        printf("%d) %s\n", i+1, exam.questions[i].question);
        for(int j=0; j<MAX_OPTIONS; j++) {
            printf("%c) %s\n", 'A'+j, exam.questions[i].options[j]);
        }
        printf("\n");
    }
    printf("**********\n");
}

// Function to grade the exam and print the results
void gradeExam(Exam exam, char answers[]) {
    exam.correctAnswers = 0;
    for(int i=0; i<exam.totalQuestions; i++) {
        if(answers[i] == exam.questions[i].answer + 'A') {
            exam.correctAnswers++;
        }
    }
    printf("**********\n");
    printf("You answered %d questions correctly out of %d.\n", exam.correctAnswers, exam.totalQuestions);
    printf("**********\n");
}

int main() {
    // Initialize the exam
    Exam exam;
    strcpy(exam.title, "C Online Exam");
    exam.totalQuestions = 3;

    // Enter the exam questions and options
    strcpy(exam.questions[0].question, "What is the output of the following code?\nint x=10; printf(\"%d\", x++); printf(\"%d\", x);");
    strcpy(exam.questions[0].options[0], "A) 1011");
    strcpy(exam.questions[0].options[1], "B) 1111");
    strcpy(exam.questions[0].options[2], "C) 1110");
    strcpy(exam.questions[0].options[3], "D) 1010");
    exam.questions[0].answer = 1;

    strcpy(exam.questions[1].question, "What is the size of int data type in bytes?");
    strcpy(exam.questions[1].options[0], "A) 2");
    strcpy(exam.questions[1].options[1], "B) 4");
    strcpy(exam.questions[1].options[2], "C) 8");
    strcpy(exam.questions[1].options[3], "D) Depends on the system architecture");
    exam.questions[1].answer = 1;

    strcpy(exam.questions[2].question, "What is the output of the following code?\nint a=5, b=10; a += (b++)+(++a); printf(\"%d %d\", a, b);");
    strcpy(exam.questions[2].options[0], "A) 27 11");
    strcpy(exam.questions[2].options[1], "B) 26 10");
    strcpy(exam.questions[2].options[2], "C) 24 12");
    strcpy(exam.questions[2].options[3], "D) Compiler error");
    exam.questions[2].answer = 0;

    // Display the exam and get user answers
    displayExam(exam);
    char userAnswers[exam.totalQuestions];
    for(int i=0; i<exam.totalQuestions; i++) {
        printf("Enter your answer for question %d: ", i+1);
        scanf(" %c", &userAnswers[i]);
    }

    // Grade the exam and print results
    gradeExam(exam, userAnswers);

    return 0;
}
