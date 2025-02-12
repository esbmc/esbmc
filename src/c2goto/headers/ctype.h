
#pragma once

#include <__esbmc/stddefs.h>

__ESBMC_C_CPP_BEGIN

int islower(int c);
int toupper(int c);
int isdigit(int c);
int isalpha(int c);
int isalnum(int c);
int isxdigit(int c);
int isupper(int c);
int tolower(int c);
int isspace(int c);
int iscntrl(int c);
int ispunct(int c);
int isprint(int c);
int isgraph(int c);
int isblank(int c);

#if defined(__GLIBC__)
// libstdc++ requires these to be defined
#ifndef _ISbit
/* These are all the characteristics of characters.
   If there get to be more than 16 distinct characteristics,
   many things must be changed that use `unsigned short int's.

   The characteristics are stored always in network byte order (big
   endian).  We define the bit value interpretations here dependent on the
   machine's byte order.  */

# include <bits/endian.h>
# if __BYTE_ORDER == __BIG_ENDIAN
#  define _ISbit(bit)	(1 << (bit))
# else /* __BYTE_ORDER == __LITTLE_ENDIAN */
#  define _ISbit(bit)	((bit) < 8 ? ((1 << (bit)) << 8) : ((1 << (bit)) >> 8))
# endif

enum
{
  _ISupper = _ISbit (0),	/* UPPERCASE.  */
  _ISlower = _ISbit (1),	/* lowercase.  */
  _ISalpha = _ISbit (2),	/* Alphabetic.  */
  _ISdigit = _ISbit (3),	/* Numeric.  */
  _ISxdigit = _ISbit (4),	/* Hexadecimal numeric.  */
  _ISspace = _ISbit (5),	/* Whitespace.  */
  _ISprint = _ISbit (6),	/* Printing.  */
  _ISgraph = _ISbit (7),	/* Graphical.  */
  _ISblank = _ISbit (8),	/* Blank (usually SPC and TAB).  */
  _IScntrl = _ISbit (9),	/* Control character.  */
  _ISpunct = _ISbit (10),	/* Punctuation.  */
  _ISalnum = _ISbit (11)	/* Alphanumeric.  */
};
#endif /* ! _ISbit  */
#endif /* __GLIBC__ */


__ESBMC_C_CPP_END
