#include <assert.h>
#include <stdint.h>

struct str { _Alignas(32) uint32_t a; };
_Static_assert(sizeof(struct str) >= 32, "clearly, 32 < 32...");

int main()
{
	struct str s, t;
	s.a = 42;
	*((uint64_t *)&s.a + 0);
}
