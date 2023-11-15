#include <string.h>
#include <stdio.h>
#include <assert.h>

typedef struct {
	int x : 12, y : (52ULL - 44);
} S;

int main()
{
	S s;
	memset(&s, 0, sizeof(s));
	printf("size: %zu\n", sizeof(s));
	assert(sizeof(s) == 4);
	s.x = -1;
	s.y = -1;
	printf("0x%08x\n", *(int *)&s);
	assert(s.y == -1);
	assert(*(int *)&s == 0x000fffff);
}
