#include "apache.h"

int ap_isspace(char c)
{
  if (c == '\t'
      || c == '\n'
      || c == '\v'
      || c == '\f'
      || c == '\r'
      || c == ' ')
    return 1;

  return 0;
}

int ap_tolower(char c)
{
  /* do we have tolower() in our stubs? */
  return c;
}

/* Rewritten to be more analyzable -- use explicit array indexing. */
char * ap_cpystrn(char *dst, const char *src, size_t dst_size)
{
  int i;

  if (dst_size == 0)
    return (dst);
  
  for (i = 0; i < dst_size - 1; i++) {
    dst[i] = src[i];
    if (src[i] == EOS) {
      return dst + i;
    }
  }

  dst[i] = EOS;

  return dst + i;
}
