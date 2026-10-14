#include <stdlib.h>

struct s
{
  int datum;
  struct s *next;
};

struct s *list;

void *safe_malloc(size_t size)
{
  void *p = malloc(size);
  if (p == 0)
    abort();
  return p;
}

int main()
{
  list = safe_malloc(sizeof(struct s));
  list->next = safe_malloc(sizeof(struct s));
  /* The only reference to the second object is dropped: a real leak, which
   * reading the buffer's contents must still report. */
  list->next = NULL;
  return 0;
}
