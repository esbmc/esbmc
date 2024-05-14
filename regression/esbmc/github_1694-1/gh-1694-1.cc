#include <cassert>

static int f(const char &x)
{
	return x;
}

int main()
{
	int v = f('b');
	assert(v == 'b');
}
