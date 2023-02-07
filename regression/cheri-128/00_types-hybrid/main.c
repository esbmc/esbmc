
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <cheri/cheric.h>

#if __WORDSIZE != 64
# error wrong architecture for cheri; is the --sysroot set correctly for mips64-unknown-linux?
#endif

#if __CHERI_PURE_CAPABILITY__ == 2
# define PTR_SZ 16
#else
# define PTR_SZ	8
#endif

int main()
{
	int a = sizeof(void * __capability);
	int b = sizeof(void *);
	int c = sizeof(intcap_t);
	int d = sizeof(uintcap_t);
	int e = sizeof(intptr_t);
	int f = sizeof(ptrdiff_t);
	int ptr_sz = PTR_SZ;
	assert(a == 16);
	assert(b == ptr_sz);
	assert(c == 16);
	assert(d == 16);
	assert(e == ptr_sz);
	assert(f == 8);
	void *__capability p = NULL;
	assert((intptr_t)p == 0);
}
