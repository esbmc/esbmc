#include "stubs.h"

#define MAX_STRING_LEN BASE_SZ + 2

int ap_isspace(char c);
int ap_tolower(char c);
char * ap_cpystrn(char *dst, const char *src, size_t dst_size);

/* GET_CHAR reads a char from a file. We're not modelling the
 * underlying file, so just non-deterministically return something. */
extern int nondet_char ();
#define GET_CHAR(c,ret) {c = nondet_char();}

