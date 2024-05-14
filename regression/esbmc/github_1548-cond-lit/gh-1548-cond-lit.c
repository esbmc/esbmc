#include <string.h>

const int b;

int main(int argc, char **argv)
{
	int *c = argc > 1 ? &b : &(int){4};
	*c = 42;
}

