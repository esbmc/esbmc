#include <stdarg.h>
#include <stdio.h>

static char out[8];

static void vp(const char *s, ...)
{
  va_list p;
  va_start(p, s);
  vprintf(s, p);
  va_end(p);
}

static void vfp(const char *s, ...)
{
  va_list p;
  va_start(p, s);
  vfprintf(stdout, s, p);
  va_end(p);
}

static void vsp(const char *s, ...)
{
  va_list p;
  va_start(p, s);
  vsprintf(out, s, p);
  va_end(p);
}

static void vsnp(const char *s, ...)
{
  va_list p;
  va_start(p, s);
  vsnprintf(out, sizeof out, s, p);
  va_end(p);
}

int main(void)
{
  char buf[1] = {'a'};
  vp("%s", buf);
  vfp("%s", buf);
  vsp("%s", buf);
  vsnp("%s", buf);
  return 0;
}
