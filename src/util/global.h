/* Global definitions for the whole project, included as required. */

#ifndef UTIL_GLOBAL_H
#define UTIL_GLOBAL_H

#ifdef _WIN32
/* On windows, define away all attributes, MSVC doesn't support them. If any are
 * any attributes to be supported cross platform, they need o have their own
 * macros. */
#define __attribute__(x)

#define snprintf _snprintf

#endif

#ifndef _WIN32
#include <alloca.h>
#endif

#endif
