void a();
void b() {
  for (;;)
    a();
}
void a() { b(); }
