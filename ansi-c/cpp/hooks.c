#include <string.h>

#include "cpp.h"

#include "../headers.h"

struct hooked_header {
	const char *basename;
	char *textstart;
	char *textend;
};

struct hooked_header headers[] = {
{ "stddef.h",		&_binary_stddef_h_start,	&_binary_stddef_h_end },
/* stddef.h contains a variety of compiler-specific functions */
{ "stdarg.h",		&_binary_stdarg_h_start,	&_binary_stdarg_h_end},
/* contains va_start and similar functionality */
{ "bits/wordsize.h",	NULL,				NULL},
/* Defines __WORDSIZE, which we define ourselves */
{ NULL, NULL, NULL}
};

int
handle_hooked_header(usch *name)
{
	struct includ buf;
	struct hooked_header *h;
	int otrulvl, c;

	fprintf(stderr, "Including file %s\n", name);

	for (h = &headers[0]; h->basename != NULL; h++) {
		if (!strcmp((char *)name, h->basename)) {
			/* This is to be hooked */
			fprintf(stderr, "Hooking...\n");

			if (h->textstart == NULL)
				return 1;

			buf.curptr = (usch*)h->textstart;
			buf.maxread = (usch*)h->textend;
			buf.buffer = (usch*)h->textstart;
			buf.infil = -1;
			buf.fname = (usch*)h->basename;
			buf.orgfn = (usch*)h->basename;
			buf.lineno = 0;
			buf.next = ifiles;
			ifiles = &buf;

			/* Largely copied from pushfile */
			if (++inclevel > MAX_INCLEVEL)
				error("Limit for nested includes exceeded");

			prtline(); /* Output file loc */

			otrulvl = trulvl;
			if ((c = yylex()) != 0)
				error("yylex returned %d", c);

			if (otrulvl != trulvl || flslvl)
				error("Unterminated conditional");

			ifiles = buf.next;
			inclevel--;
			return 1;
		}
	}

	return 0;
}
