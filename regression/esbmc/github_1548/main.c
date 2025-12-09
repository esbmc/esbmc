#include <string.h>

const int b;

int main(int argc, char **argv)
{
	int *c = &b;
	*c = 42;
}

