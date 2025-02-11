#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

int main()
{
	double x = -0.0;
	union {
		uint64_t k;
		double y;
	} u = { 0 };
	assert(x == u.y);
	int k = memcmp(&x, &u.y, sizeof(x));
	assert(k != 0);
}
