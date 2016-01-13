#include <string.h>

int main(void)
{
    char *src = "Take the test.";
//  src[0] = 'M' ; // this would be undefined behavior
    char dst[16]; // +1 to accomodate for the null terminator
    strcpy(dst, src);
    dst[0] = 'M'; // OK

    assert(strcmp(dst, src));
}
