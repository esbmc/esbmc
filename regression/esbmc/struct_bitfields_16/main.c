#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <assert.h>

void print_bits(uint32_t k)
{
	size_t n = sizeof(k)*CHAR_BIT;
	for (size_t i=0; i<n;) {
		printf("%c", (k >> (n - i - 1)) & 1 ? '1' : '0');
		if (++i % CHAR_BIT == 0)
			printf(" ");
	}
	printf("= %u\n", k);
}

int main(int argc, char **argv)
{
	union {
		struct {
			unsigned int s : 5;
			unsigned short f : 7;
			unsigned char : 2;
			unsigned int c : 3;
		};
		uint32_t i;
	} v = { .c = 1 << 2 | 1, .s = 1 << 4 | 1, .f = 1 << 6 | 1 };
#ifndef __clang__
	print_bits(v.i);
#endif
        assert(v.i == 84017);
}
