#include <stdlib.h>

void foo(void *p)
{
	return free(p);
}

int main()
{
	foo(malloc(5));
}
