
#include <assert.h>
#include <stdint.h>

int arr[16];

void f(unsigned k, unsigned v)
{
	k %= 16;
	arr[k] = v;
	void *p = arr;
	int w = *(int *)(void *)((short *)arr + (k * 2));
	assert(w == v);
}

int main(int argc, char **argv)
{
	f(atoi(argv[1]), atoi(argv[2]));
}
