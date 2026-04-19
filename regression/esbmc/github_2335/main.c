#include <stdlib.h>

struct s {
  int datum;
  struct s *next;
};

struct s *slot[2];

struct s *new(int x) {
  struct s *p = malloc(sizeof(struct s));
  p->datum = x;
  p->next = ((void *)0);
  return p;
}
void list_add(struct s *node, struct s *list) {
  struct s *temp = list->next;
  list->next = node;
  node->next = temp;
}
int main () {
  slot[0] = new(1);
  list_add(new(2), slot[0]);
  slot[1] = new(1);
  list_add(new(2), slot[1]);
  list_add(new(3), slot[1]);
  return 0;
}
