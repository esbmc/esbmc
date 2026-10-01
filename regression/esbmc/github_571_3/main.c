#include <assert.h>

/* The big-endian layout from #571: --big-endian does not change clang's
   __BYTE_ORDER__, so the declaration cannot be chosen by it. */

struct S {
	unsigned a : 12;
	unsigned b : 12;
};

int main()
{
	union {
		unsigned sh;
		struct S s;
	} u = { 0x00123456 };
	assert(u.s.a == 0x001);
	assert(u.s.b == 0x234);
}
