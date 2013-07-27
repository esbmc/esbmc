#include <string.h>

#include "cpp.h"

#include "../headers.h"

struct hooked_header {
	const char *basename;
	char *textstart;
	unsigned int *textsize;
};

struct hooked_header headers[] = {
{ "stddef.h",		stddef_buf,	&stddef_buf_size},
/* stddef.h contains a variety of compiler-specific functions */
{ "stdarg.h",		stdarg_buf,	&stdarg_buf_size},
/* contains va_start and similar functionality */
{ "stdbool.h",		stdbool_buf,	&stdbool_buf_size},
/* Fairly self explanatory */
{ "bits/wordsize.h",	NULL,				NULL},
/* Defines __WORDSIZE, which we define ourselves */
{ "pthread.h",		pthread_buf,	&pthread_buf_size},
/* Pthreads header */
{ "digitalfilter.h",    digitalfilter_buf, &digitalfilter_buf_size},
/* digital filter header */
{ "pthreadtypes.h",	pthreadtypes_buf, &pthreadtypes_buf_size},
/*  Additional pthread data header */
{ NULL, NULL, NULL}
};

#undef p

int
handle_hooked_header(usch *name)
{
	struct includ buf;
	struct hooked_header *h;
	int otrulvl, c;

	for (h = &headers[0]; h->basename != NULL; h++) {
		if (!strcmp((char *)name, h->basename)) {
			/* This is to be hooked */

			if (h->textstart == NULL)
				return 1;

			buf.curptr = (usch*)h->textstart;
			buf.maxread = (usch*)h->textstart + *h->textsize;
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
			if ((c = cpplex()) != 0)
				error("cpplex returned %d", c);

			if (otrulvl != trulvl || flslvl)
				error("Unterminated conditional");

			ifiles = buf.next;
			inclevel--;
			return 1;
		}
	}

	return 0;
}
