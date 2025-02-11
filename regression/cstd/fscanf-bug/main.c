#include <stdio.h>
#include <assert.h>
int main(int argc, char * argv[])
{
    int m, n = 42;
    fscanf(stdin, "%d", &m, &n);
    assert(n == 42);
}
