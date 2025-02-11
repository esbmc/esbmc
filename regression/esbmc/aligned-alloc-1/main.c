
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <stdint.h>

void * aligned_alloc(size_t, size_t);

int main()
{
	size_t align = nondet_uint();
	size_t size = nondet_uint();
	void *r = aligned_alloc(align, size);
	assert(!r || !align || !(size & (align - 1)));
	assert(!((uintptr_t)r & (align - 1)));
	free(r);
}
