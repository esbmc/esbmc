#include <stdlib.h>
#include <assert.h>

void steal_addr_space(size_t n)
{
        char *a = malloc(n);
        assert(__ESBMC_get_object_size(a) == n);
        free(a);
}

int main()
{
        size_t GiB = 1ULL << 30;
        for (int i=0; i<4; i++)
                steal_addr_space(1 * GiB);
        assert(0);
}
