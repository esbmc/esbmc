#include <stdlib.h>

int main(void)
{
    float *buf = malloc(4 * sizeof(float));
    if (!buf) exit(1);

    char *p = (char *)buf;
    ++p;
    float *misaligned_float_ptr = (float *)p;
    *misaligned_float_ptr = 42.0;

    return 0;
}
