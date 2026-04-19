#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
    char name[20] = "";

    __ESBMC_assume(argc == 10);
    __ESBMC_assert(0, "0");
}
