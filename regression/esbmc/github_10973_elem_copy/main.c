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

static void g(va_list outer, const char *s, ...)
{
  COPY(q, outer);
  vprintf(s, LIST(q));
}

static void f(const char *s, ...)
{
  va_list p;
  va_start(p, s);
  g(p, s, (char *)0);
  va_end(p);
}

int main(void)
{
  f("%s", "ok");
  return 0;
}
