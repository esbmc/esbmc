void a();
b() {
  for (;;)
    a();
}
void a() { b(); }
