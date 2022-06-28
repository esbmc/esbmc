#include <string.h>

typedef struct __attribute__((__packed__)) {
	char a;
	short b;
} P;

static P q[17];

short fun1(size_t i)
{
	__ESBMC_assume(i < 17);
	return q[i].b;
}
