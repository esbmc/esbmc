#include <string.h>
#include <assert.h>

int main()
{
	int no = __VERIFIER_nondet_int();
	const char *err = strerror(no);
	assert(err); // C99 says this must not be NULL
}
