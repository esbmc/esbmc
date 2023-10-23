#include <math.h>
#include <assert.h>

int main()
{
	int e;
	double r = frexp(INFINITY, &e);
	assert(isinf(r));
}
