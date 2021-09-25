#include "stubs.h"

/* Vary these to affect the analysis difficulty of the variants
 * calling strncmp() */
#define LDAP "ldap"
#define LDAP_SZ 4

/* Size of the buffer being overflowed 
 * Must ensure that 0 < TOKEN_SZ - 1 */
#define TOKEN_SZ BASE_SZ + 1

/* This requires an explanation. escape_absolute_uri() gets passed a
 * buffer uri[] and an offset into uri[]. The loop which overflows
 * token[] is only executed if uri[] starts with the string LDAP of
 * size LDAP_SZ, and if the character in uri[] which is one past the
 * offset is a slash. Hence the LDAP_SZ (for the string LDAP) and the
 * first +1 (for the slash).
 *
 * The second +1 is because we increment our iterator over uri[] at
 * least once before reaching the loop which overflows token[].
 *
 * The TOKEN_SZ + 2 is there so that uri[] will have enough characters
 * after the offset to overflow token[].
 */
#define URI_SZ LDAP_SZ + 1 + 1 + TOKEN_SZ + 2

