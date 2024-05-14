float a;
main() {
  for (;;) {
    memset(&a, 0, sizeof(a)-1);
    reach_error();
    for (;;)
      ;
  }
}
