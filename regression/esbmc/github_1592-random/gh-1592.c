#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

int main()
{
	int v = random();
	assert(v != INT32_MAX);
}
