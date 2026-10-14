/* A byte view reached through a pointer is rooted at the pointee, not at the
 * pointer variable. */
struct S { int x; int y; };
struct S s;
int main(void) {
  struct S *p = &s;
  int d = (char *)&p->x == (char *)&p; /* 0: p->x is s.x, not the variable p */
  return 10 / ((char *)&p->x == (char *)&p) + d;
}
