#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

void steal_addr_space(size_t n)
{
	char *a = malloc(n);
	assert(__ESBMC_get_object_size(a) == n);
	// free(a);
}

int main()
{
	/* Two objects at the per-object cap, which together still exhaust a 32-bit
	   address space. Was 2 GiB each; allocation is now capped at PTRDIFF_MAX
	   (2 GiB - 1 here), so a 2 GiB request returns NULL and never reached the
	   exhaustion this test is about. */
	for (int i=0; i<2; i++)
		steal_addr_space(PTRDIFF_MAX);
	assert(0);
}
