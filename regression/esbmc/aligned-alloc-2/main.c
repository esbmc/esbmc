
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <stdint.h>

void * aligned_alloc(size_t, size_t);

typedef struct {
	_Alignas(16)
	uint64_t x;
} a;

int main()
{
	size_t align = 16;
	size_t size = 32;
	a *r = aligned_alloc(align, size);
	assert(r);
	r[0].x = 0;
	r[1].x = 1;
	free(r);
}
