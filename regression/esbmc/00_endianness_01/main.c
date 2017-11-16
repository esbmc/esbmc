#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 1//64

void myMemcpy(void *dst, const void *src,	size_t count)
{
	char *cdst = (char *) dst;
	const char *csrc = (const char *)src;
	int numbytes = count/(sizeof(char));
	for (int i = 0; i < numbytes; i++)
		cdst[i] = csrc[i];
}

int f(int x) {
	
	return x + 2;
}

void foo(int *y, int x) {

	*y = f(x);

}

int main() {
	int a=2;
	int b=0;
	int *dev_a;

	dev_a = (int*) malloc(N*sizeof(int));

	myMemcpy(dev_a, &a, sizeof(int));

	foo(dev_a, a);

	myMemcpy(&b, dev_a, sizeof(int));

	printf("%d", b);
	assert (b == a+2); 

	free(dev_a);

	return 0;
}
