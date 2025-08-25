#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main() {
    char *str1 = malloc(sizeof(char)*3);
    char *str2 = malloc(sizeof(char)*3);

    strcpy(str1, "11");
    strcpy(str2, "11");

    assert(strcmp(str1, str2) == 0);

    return 0;
}
