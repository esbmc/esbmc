#include <assert.h>

#define NULL 0
#define TYPE int
#define INTR int

TYPE a;
TYPE *dev_a;
	
int main()
{
	dev_a = (int*) operator new(sizeof(int));
	assert(dev_a != NULL);
	assert(*dev_a == 0);

	return 0;
}
