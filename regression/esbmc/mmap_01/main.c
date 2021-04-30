#include <stdio.h>
#include <sys/mman.h>

int main(int argc, char *argv[])
{
	int *p;
	unsigned int n  = nondet_uint();
	
	__ESBMC_assume(n > 1);
	__ESBMC_assume(n < 10000000);

	p = mmap(NULL, n*sizeof(int), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
	__ESBMC_assume(p);	

	return 0;
}

