#include <string.h>

int main(int argc, char **argv)
{
	static const char *b[3];

	char *c = (void *)b;
	*c = 42;
}

