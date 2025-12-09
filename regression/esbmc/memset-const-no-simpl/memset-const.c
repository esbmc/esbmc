#include <string.h>

const char b[] = "abc";
const char *d = "xyz";

int main(int argc, char **argv)
{
	char a[2];
	char *c = argc == 1 ? a : argc == 2 ? b : d;
	memset(c, 0, 1);
}
