#include <limits.h>

struct aws_string
{
  const unsigned long len;
  const unsigned char bytes[];
};

void *bounded_malloc(unsigned long size)
{
  __ESBMC_assume(size <= (ULONG_MAX >> (8 + 1)));
  return malloc(size);
}

static inline const unsigned char *
aws_string_bytes(const struct aws_string *str)
{
  return str->bytes;
}

int main()
{
  unsigned long len;
  __ESBMC_assume(
    len < (ULONG_MAX - 1 - sizeof(struct aws_string)));

  struct aws_string *str = bounded_malloc(sizeof(struct aws_string) + len + 1);
  *(unsigned long *)(&str->len) = len;
  *(unsigned char *)&str->bytes[len] = '\0';

  __ESBMC_assert(aws_string_bytes(str) == str->bytes, "This should never fail");

  __ESBMC_assert(
    ((((str->len + 1)) == 0) || ((&str->bytes[0]))), "This should never fail");
  __ESBMC_assert(str->bytes[str->len] == 0, "This should never fail");
}
