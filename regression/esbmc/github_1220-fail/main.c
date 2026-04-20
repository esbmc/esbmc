#include <assert.h>

int arr[16];

void f(unsigned k, unsigned v)
{
	k %= 16;
	arr[k] = v;
	void *p = arr;
	int *q = (int *)(void *)((short *)p + (k * 1));
	int w = *q;
	assert(w == v);
}
