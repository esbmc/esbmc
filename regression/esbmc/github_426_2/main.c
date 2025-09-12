#include <stdint.h>
#include <assert.h>
#include <stddef.h>
#include <limits.h>

int main(void) {
	struct S { int x; int y; int z; } s = { .z = 42 };
	uintptr_t v = (uintptr_t)&s.x;
	uintptr_t u = offsetof(struct S, y);
	u *= 2;
	int *p = (int *)(u + v);
	*p = 3;
	assert(&s.z == p);
	assert(s.z == 3);
}
