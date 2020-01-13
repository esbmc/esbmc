#include <string.h>
#include <stdlib.h>

#include "cpp.h"
#include "iface.h"

#include "../headers.h"

extern int inclevel; // Good grief

struct hooked_header {
	const char *basename;
	char *textstart;
	unsigned int *textsize;
};

struct hooked_header headers[] = {
/* stddef.h contains a variety of compiler-specific functions */
{ "stddef.h",		stddef_buf,	&stddef_buf_size},
/* contains va_start and similar functionality */
{ "stdarg.h",		stdarg_buf,	&stdarg_buf_size},
/* Fairly self explanatory */
{ "stdbool.h",		stdbool_buf,	&stdbool_buf_size},
/* Defines __WORDSIZE, which we define ourselves */
{ "bits/wordsize.h",	NULL,				NULL},
/* Pthreads header */
{ "pthread.h",		pthread_buf,	&pthread_buf_size},
/*  Additional pthread data header */
{ "pthreadtypes.h",	pthreadtypes_buf, &pthreadtypes_buf_size},
 /* Integer limits */
{ "limits.h",		limits_buf, &limits_buf_size},
{ NULL, NULL, NULL}
};

#undef p

int
handle_hooked_header(const usch *name)
{
	struct includ buf;
	struct hooked_header *h;
	int otrulvl, c;

	for (h = &headers[0]; h->basename != NULL; h++) {
		if (!strcmp((char *)name, h->basename)) {
			/* This is to be hooked */

			if (h->textstart == NULL)
				return 1;

			// Due to some horror, it looks like there needs to
			// be a leading lump of buffer space ahead of the text
			// being parsed.
			buf.bbuf = malloc(*h->textsize + NAMEMAX);
                        memcpy(buf.bbuf + NAMEMAX, h->textstart, *h->textsize);
			buf.curptr = buf.bbuf + NAMEMAX;
			buf.buffer = buf.curptr;
			buf.maxread = buf.curptr + *h->textsize;
			buf.infil = -1;
			buf.fname = (usch*)h->basename;
			buf.fn = (usch*)h->basename;
			buf.orgfn = (usch*)h->basename;
			buf.lineno = 0;
			buf.escln = 0;
			buf.next = ifiles;
			buf.idx = SYSINC;
			buf.incs = NULL;
			ifiles = &buf;

			/* Largely copied from pushfile */
			if (++inclevel > 100)
				error("Limit for nested includes exceeded");

			prtline(); /* Output file loc */

			otrulvl = trulvl;

                        fastscan();

			if (otrulvl != trulvl || flslvl)
				error("Unterminated conditional");

			ifiles = buf.next;
			inclevel--;
			return 1;
		}
	}

	return 0;
}
