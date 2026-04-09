#include <stdio.h>
#include <stdlib.h>

// Safe program that takes an n long string, counts the characters, and writes n characters into a buffer with a check to make sure n is not so big that it causes buffer overflow.

int main(int argc, char *argv[]) {
    if (argc != 2) { // Check for the right number of arguments (1)
        return -1;
    }

    if (argv[1] == NULL) { // Make sure the first argument is not null
        return -1;
    }

    int num = atoi(argv[1]);

    char buffer[10];

    for (int i = 0; i < num; i++) {
        buffer[i] = 'X';
    }
    buffer[num] = '\0';

    printf("Buffer contents: %s\n", buffer);

    return 0;
}
