#include <math.h>
#include <assert.h>

int main()
{
	double inf = 1.0 / 0.0;
	assert(inf == INFINITY);
	assert(!isfinite(inf));
}
