#include <assert.h>

#define NULL 0
#define TYPE int
#define INTR int

TYPE a;
TYPE *dev_a;
	
int main()
{
	dev_a = new int();
	assert(dev_a != NULL);
	assert(*dev_a == 0);

	int a = 10;
	dev_a = &a;
	assert(*dev_a == 10);

	return 0;
}
