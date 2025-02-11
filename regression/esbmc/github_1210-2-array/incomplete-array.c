
#include <assert.h>
#include <string.h>

extern int JJ[];

int main()
{
	int k = 42;
	memcpy(JJ, &k, sizeof(k));
	int j;
	memcpy(&j, JJ, sizeof(k));
	assert(j == k);
}
