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
		struct S s;
		unsigned short sh;
	} u = { { .a = 1, .b = 2, .c = 3, .d = 4 } };
	assert(u.sh == 0x1234);
}
