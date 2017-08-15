#include <assert.h>

int
foobar(int asdf)
{
	return asdf;
}

int
main()
{
	assert(foobar(0) == 1);
	return 0;
}
