#include <string.h>

#include "cpp.h"

#include "../headers.h"

struct hooked_header {
	const char *basename;
	char *textstart;
	char *textend;
};

/* Drama: when building with mingw, an additional '_' character is placed at the
 * beginning of all symbols. Wheras the header objects produced by ld in
 * ansi-c/headers will only ever have one '_' character at the start. So, some
 * hackery is required */

#if defined(__MINGW32__) && !defined(__MINGW64_VERSION_MAJOR)
#define p(x) x
#else
#define p(x) _##x
#endif

struct hooked_header headers[] = {
{ "stddef.h",		&p(binary_stddef_h_start),	&p(binary_stddef_h_end) },
/* stddef.h contains a variety of compiler-specific functions */
{ "stdarg.h",		&p(binary_stdarg_h_start),	&p(binary_stdarg_h_end)},
/* contains va_start and similar functionality */
{ "stdbool.h",		&p(binary_stdbool_h_start),	&p(binary_stdbool_h_end)},
/* Fairly self explanatory */
{ "bits/wordsize.h",	NULL,				NULL},
/* Defines __WORDSIZE, which we define ourselves */
{ "pthread.h",		&p(binary_pthread_h_start),	&p(binary_pthread_h_end)
},
/* Pthreads header */
{ "pthreadtypes.h",	&p(binary_pthreadtypes_h_start),&p(binary_pthreadtypes_h_end)
},
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
