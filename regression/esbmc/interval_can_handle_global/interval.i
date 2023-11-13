c, e = 1;
a();
void b();
void d();
main() {
  c = 1;
  if (c == 1) {
    b();
    (*a)();
  }
  d();
}
void b() { 
int tmp = e;
e = tmp + 1; }
void d() {
  if (e != 1)
    reach_error();
}
