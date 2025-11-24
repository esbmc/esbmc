#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
    char name[10];

    // Accessing argv[1] without checking argc
    strcpy(name, argv[1]);  // Buffer overflow if input is too long!

    return 0;
}

