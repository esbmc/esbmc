#include <string.h>

typedef struct __attribute__((__packed__)) {
	char a;
	short b;
} P;

static P q[17];

short fun2(P *p, size_t i)
{
	__ESBMC_assume(i < 17);
	return p[i].b;
}
