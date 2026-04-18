#include <assert.h>

struct S {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
	unsigned d : 4;
	unsigned c : 4;
	unsigned b : 4;
	unsigned a : 4;
#else
	unsigned a : 4;
	unsigned b : 4;
	unsigned c : 4;
	unsigned d : 4;
#endif
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
