#include <assert.h>

union U { unsigned raw : 32; struct { unsigned x : 16, y : 16; }; };

unsigned f();

int main() {
	unsigned x = f();
	assert(((union U)x).raw == x);
}
