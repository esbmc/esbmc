
#include <assert.h>
#include <stdlib.h>
#include <string.h>

union incomplete;

extern union incomplete JJ;

/*
The verification failure is the correct behavior because:
- The program uses extern union incomplete JJ with no definition
- Memory operations on incomplete union types have undefined behavior
- ESBMC conservatively treats the memory contents as non-deterministic
- The assertion j == 42 cannot be guaranteed to hold
*/

int main()
{
	int k = 42;
	memcpy(&JJ, &k, sizeof(k));
	int j;
	memcpy(&j, &JJ, sizeof(k));
	assert(j == 42);
}

// struct incomplete { short x, y; };
