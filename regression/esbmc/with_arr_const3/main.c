#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main()
{
    char str1[2];
    char str2[2];

    strcpy(str1, "1");
    strcpy(str2, "1");

    assert(strcmp(str1, str2) == 0);

    return 0;
}
