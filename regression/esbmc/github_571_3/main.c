#include <assert.h>

struct S {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
	unsigned b : 12;
	unsigned a : 12;
#else
	unsigned a : 12;
	unsigned b : 12;
#endif
};

int main()
{
	union {
		unsigned sh;
		struct S s;
	} u = { 0x00123456 };
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
	assert(u.s.a == 0x123);
	assert(u.s.b == 0x456);
#else
	assert(u.s.a == 0x001);
	assert(u.s.b == 0x234);
#endif
}
