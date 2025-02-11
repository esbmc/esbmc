//FormAI DATASET v0.1 Category: Cat Language Translator ; Style: excited
// Welcome to the C Cat Language Translator!
// This program will translate any phrase into C Cat Language!

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to translate a given phrase into C Cat Language
char* C_CAT(char* phrase) {
    int len = strlen(phrase);
    char* c_cat_phrase = (char*)malloc(sizeof(char) * (len + 1)); // Allocate memory for c_cat_phrase
    strcpy(c_cat_phrase, phrase); // Copy original phrase to c_cat_phrase
    
    // Loop through c_cat_phrase and replace vowels with "meow"
    for (int i = 0; i < len; i++) {
        if (c_cat_phrase[i] == 'a' || c_cat_phrase[i] == 'e' || c_cat_phrase[i] == 'i' || c_cat_phrase[i] == 'o' || c_cat_phrase[i] == 'u' ||
            c_cat_phrase[i] == 'A' || c_cat_phrase[i] == 'E' || c_cat_phrase[i] == 'I' || c_cat_phrase[i] == 'O' || c_cat_phrase[i] == 'U') {
            c_cat_phrase[i] = 'm';
            c_cat_phrase = (char*)realloc(c_cat_phrase, sizeof(char) * (len + 3)); // Reallocate memory for extra characters
            strcat(c_cat_phrase, "eow"); // Add "eow" after vowel
            len += 3;
        }
    }
    
    return c_cat_phrase;
}

int main() {
    // Welcome message
    printf("Welcome to the C Cat Language Translator!\n\n");
    
    // Prompt user for phrase to translate
    char phrase[100];
    printf("Please enter a phrase to translate into C Cat Language: ");
    fgets(phrase, 100, stdin);
    
    // Remove trailing newline character
    phrase[strlen(phrase) - 1] = '\0';
    
    // Translate phrase into C Cat Language
    char* c_cat_phrase = C_CAT(phrase);
    
    // Print translated phrase
    printf("\n%s\n", c_cat_phrase);
    
    // Free memory used by c_cat_phrase
    free(c_cat_phrase);
    
    // Closing message
    printf("Thank you for translating with the C Cat Language Translator!\n");
    
    return 0;
}
