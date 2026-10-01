#include <assert.h>

/* The big-endian layout from #571: --big-endian does not change clang's
   __BYTE_ORDER__, so the declaration cannot be chosen by it. */

struct S {
	unsigned a : 4;
	unsigned b : 4;
	unsigned c : 4;
	unsigned d : 4;
};

int main()
{
	union {
		unsigned short sh;
		struct S s;
	} u = { 0x1234 };
	assert(u.s.a == 1);
	assert(u.s.b == 2);
	assert(u.s.c == 3);
	assert(u.s.d == 4);
}
