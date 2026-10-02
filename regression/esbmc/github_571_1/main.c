#include <assert.h>

/* The big-endian layout from #571. */

struct S {
	unsigned a : 4;
	unsigned b : 4;
	unsigned c : 4;
	unsigned d : 4;
};

int main()
{
	union {
		struct S s;
		unsigned short sh;
	} u = { { .a = 1, .b = 2, .c = 3, .d = 4 } };
	assert(u.sh == 0x1234);
}
