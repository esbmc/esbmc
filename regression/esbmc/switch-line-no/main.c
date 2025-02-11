#include <assert.h>

int main()
{
	int data = 0;
	switch (42) {
	case 42: { // line 7
		data = nondet_int(); // line 8
		break;
	}
	default:
		break;
	}
	assert(!data);
}
