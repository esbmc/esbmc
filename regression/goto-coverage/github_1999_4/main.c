#include <assert.h>
#include <stdbool.h>

bool t2()
{
	bool x;
	return x + 1;
}

bool t1()
{
	return 1 == 2 && t2();
}

void test(bool x)
{
}

int main()
{
	int nS;
	int nE = t1();
	return ((nS != nE) && t1() || (2 == 3));
}
