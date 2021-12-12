
#include <stdint.h>
#include <assert.h>
#include <stddef.h>

int main(void) {
	struct S { int x; int y; int z; } s = { .x = 42 };
	uintptr_t u = (uintptr_t)&s;
	u *= 2;
	u -= (uintptr_t)&s;
	int *p = (int *)(u);
	*p = 3;
	assert(&s.x == p);
	assert(*p < 0 || *p == 0 || *p > 0);
	assert(s.x == 3);
}
