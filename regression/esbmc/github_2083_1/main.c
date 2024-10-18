#include <assert.h>

void foo(unsigned int x)
{
	x++;
	if ((10 * x - 1) == 1001)
		assert(0);
}

int main()
{
	foo(3435973937); // hold
}
