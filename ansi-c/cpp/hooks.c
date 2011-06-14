#include <string.h>

#include "cpp.h"

#include "../headers.h"

struct hooked_header {
	const char *basename;
	void *textstart;
	void *textend;
};

struct hooked_header headers[] = {
{ "stddef.h",		&_binary_stddef_c_start,	&_binary_stddef_c_end },
/* stddef.h contains a variety of compiler-specific functions */
{ NULL, NULL, NULL}
};

int
handle_hooked_header(usch *name)
{

	fprintf(stderr, "Including file %s\n", name);
	return 0;
}
