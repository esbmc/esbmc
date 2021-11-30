
#include <stdint.h>
#include <assert.h>
#include <stddef.h>
#include <limits.h>

int main(void) {
	// struct { char c; int y; int z; } s = { .y = 42 };
	int a[3] = { [1] = 42 };
	uintptr_t u = (uintptr_t)&a[0];
	u += 4;
	int *p = (int *)(u);
	*p = 3;
	assert(a[1] == 3);
}
