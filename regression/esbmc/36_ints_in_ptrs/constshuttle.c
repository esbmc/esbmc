#include <stdbool.h>

bool nondet_bool();

int
main()
{
	void *face, *bees;
	int num;

	bees = &num;
	face = (void*)1234;
	num = (int)face;

	assert(num == 1234);

	face = (nondet_bool()) ? (void*)1234 : (void*)1234;
	num = (int)face;

        assert(num == 1234);

	return 0;
}
