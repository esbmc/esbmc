
#include <stdlib.h>
#include <assert.h>
#include <errno.h>

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
