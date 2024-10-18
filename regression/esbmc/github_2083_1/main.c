#include <assert.h>

void foo(unsigned int x)
{
	x++;
	assert(x > 3);
}

int main()
{
	foo(3435973937); // hold
}
