#include <stdlib.h>
#include <assert.h>

void steal_addr_space(size_t n)
{
	char *a = malloc(n);
	assert(a);
	// free(a);
}

int main()
{
	size_t MiB = 1ULL << 20;
	/* 16 * 256 MiB = 4 GiB */
	for (int i=0; i<16; i++)
		steal_addr_space(256 * MiB);
	assert(0);
}
