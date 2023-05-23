
#include <stdint.h>
#include <assert.h>

typedef union {
	uint8_t a[2][4];
	uint64_t u64;
} U;

U init()
{
	U r;
	r.u64 = 0x1122334455667788ULL;
	return r;
}

int main()
{
	U u = init();
	assert(u.a[1][0] == 0x44);
	assert(u.a[1][1] == 0x33);
	assert(u.a[1][2] == 0x22);
	assert(u.a[1][3] == 0x11);
}
