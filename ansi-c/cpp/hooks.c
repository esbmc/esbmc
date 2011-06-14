#include "cpp.h"

#include <string.h>

struct hooked_header {
	const char *basename;
	void *textstart;
	void *textend;
};

struct hooked_header headers[] = {
{ "stddef.h",		NULL,		NULL }
/* stddef.h contains a variety of compiler-specific functions */
};

int
handle_hooked_header(usch *name)
{

	fprintf(stderr, "Including file %s\n", name);
	return 0;
}
