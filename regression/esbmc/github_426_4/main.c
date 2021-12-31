
#include <stdint.h>
#include <assert.h>
#include <stddef.h>

int main(void) {
	struct S { int x; int y; int z; } s = { .z = 42 };
	uintptr_t u = (uintptr_t)&s.y;
	u *= 2;
	u -= (uintptr_t)&s.x;
	int *p = (int *)(u);
	*p = 3;
	assert(s.z == 3);
}
