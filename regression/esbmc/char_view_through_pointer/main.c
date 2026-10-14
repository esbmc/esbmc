/* A byte view reached through a pointer is rooted at the pointee, not at the
 * pointer variable. */
struct S { int x; int y; };
struct S s;
int main(void) {
  struct S *p = &s;
  return 10 / ((char *)&p->y == (char *)&s.y); /* divisor 1, no UB */
}
