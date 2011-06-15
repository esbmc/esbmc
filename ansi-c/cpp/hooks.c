#include <string.h>

#include "cpp.h"

#include "../headers.h"

struct hooked_header {
	const char *basename;
	char *textstart;
	char *textend;
};

struct hooked_header headers[] = {
{ "stddef.h",		&_binary_stddef_c_start,	&_binary_stddef_c_end },
/* stddef.h contains a variety of compiler-specific functions */
{ "stdarg.h",		&_binary_stdarg_c_start,	&_binary_stdarg_c_end},
/* contains va_start and similar functionality */
{ NULL, NULL, NULL}
};

int
handle_hooked_header(usch *name)
{
	struct hooked_header *h;
	unsigned long int i, length;

	fprintf(stderr, "Including file %s\n", name);

	for (h = &headers[0]; h->basename != NULL; h++) {
		if (!strcmp((char *)name, h->basename)) {
			/* This is to be hooked */
			fprintf(stderr, "Hooking...\n");

			length = h->textend - h->textstart;
			for (i = 0; i < length; i++) {
				putch(h->textstart[i]);
			}

			return 1;
		}
	}

	return 0;
}
