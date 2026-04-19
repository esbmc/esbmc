float a;
main() {
  for (;;) {
    memset(&a, 0, sizeof(a));
    reach_error();
    for (;;)
      ;
  }
}
