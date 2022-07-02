#include <stdlib.h>
#include <assert.h>

int main()
{
	size_t GiB = 1ULL << 30;
	if (malloc(2 * GiB))
		assert(!malloc(2 * GiB));
}
