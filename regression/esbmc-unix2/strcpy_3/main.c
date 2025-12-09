#include <stdio.h>
#include <string.h>
#include <assert.h>

void test_strcpy() {
    char *source=NULL;
    char destination[] = "Hello World!";

    // Copy source string to destination
    strcpy(destination, source);

    // this assertion is reacbale
    assert(0);

    // Check if the strings are the same
    if (strcmp(source, destination) == 0) {
        printf("Test Passed: The strings are identical.\n");
    } else {
        printf("Test Failed: The strings are different.\n");
    }

    // Print the copied string
    printf("Source: %s\n", source);
    printf("Destination: %s\n", destination);
}

int main() {
    // Run the test case
    test_strcpy();
    return 0;
}


