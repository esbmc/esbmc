/* This file defines pointers to the locations in ESBMC of headers-as-object
 * files */

/* There's additional pain in that the number of '_' characters at the start of
 * the symbol varies when compiling for windows, but doesn't when using LD to
 * produce the header blobs. So, hacks: */

#ifdef __MINGW32__
#define p(x) (x)
#else
#define p(x) _##x
#endif

extern char p(binary_stddef_h_start);
extern char p(binary_stddef_h_end);
extern char p(binary_stdarg_h_start);
extern char p(binary_stdarg_h_end);
extern char p(binary_stdbool_h_start);
extern char p(binary_stdbool_h_end);
extern char p(binary_pthread_h_start);
extern char p(binary_pthread_h_end);
extern char p(binary_pthreadtypes_h_start);
extern char p(binary_pthreadtypes_h_end);

#undef p
