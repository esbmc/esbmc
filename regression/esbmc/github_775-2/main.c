#include <stdlib.h>
#include <assert.h>

void steal_addr_space(size_t n)
{
	char *a = malloc(n);
	// free(a);
}

int main()
{
	size_t GiB = 1ULL << 30;
	/* 2 * 2 GiB = 4 GiB */
	for (int i=0; i<2; i++)
		steal_addr_space(2 * GiB);
	assert(0);
}
