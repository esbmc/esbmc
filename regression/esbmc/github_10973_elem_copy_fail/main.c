#include <assert.h>
#include <stdarg.h>
#include <stdio.h>

// q copies the caller's va_list element by element, so q denotes the
// caller's arguments, not g's own.
#if defined(__x86_64__)
#define COPY(q, outer) va_list q; *q = *outer
#define LIST(q) q
#else
#define COPY(q, outer) struct { va_list v; } q; q.v = outer
#define LIST(q) q.v
#endif

static int g(va_list outer, const char *s, ...)
{
  COPY(q, outer);
  return vsnprintf(NULL, 0, s, LIST(q));
}

static int f(const char *s, ...)
{
  va_list p;
  va_start(p, s);
  int r = g(p, s, "x");
  va_end(p);
  return r;
}

int main(void)
{
  assert(f("%s", "longer than four") < 5);
  return 0;
}
