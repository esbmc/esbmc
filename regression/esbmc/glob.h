#include "stubs.h"

/* (One less than the) size of the buffer being overflowed. */
#define MAXPATHLEN BASE_SZ

/* Make PATTERNLEN bigger than MAXPATHLEN -- we want to be able to overflow 
 * the buffer of length MAXPATHLEN+1 without having a tool complain about 
 * out-of-bounds reads of the pattern buffer.
 */
#define PATTERNLEN MAXPATHLEN+5

/* Size of d_name. We don't care about it; like PATTERNLEN, just make
 * it "big enough".
 */
#define MAXNAMLEN (MAXPATHLEN * 4)

#define	DOLLAR		'$'
#define	DOT		'.'
#define	LBRACKET	'['
#define	NOT		'!'
#define	QUESTION	'?'
#define	QUOTE		'\\'
#define	RANGE		'-'
#define	RBRACKET	']'
#define	SEP		'/'
#define	STAR		'*'
#define	TILDE		'~'
#define	UNDERSCORE	'_'
#define	LBRACE		'{'
#define	RBRACE		'}'
#define	SLASH		'/'
#define	COMMA		','

#define	M_QUOTE		0x80
#define	M_PROTECT	0x40
#define	M_MASK		0xff
#define	M_ASCII		0x7f

/* In the original, a Char is an unsigned short.
 *
 * However, this triggers a bug in SatAbs. Hence, it's an int.
 */
//typedef unsigned short Char;
typedef int Char;
typedef char u_char;

#define	CHAR(c)		((Char)((c)&M_ASCII))
#define	META(c)		((Char)((c)|M_QUOTE))
#define	M_ALL		META('*')
#define	M_END		META(']')
#define	M_NOT		META('!')
#define	M_ONE		META('?')
#define	M_RNG		META('-')
#define	M_SET		META('[')
#define	ismeta(c)	(((c)&(0x80)) != 0)

#define GLOB_ABORTED -1

// For SatAbs
extern int nondet_int (void);
