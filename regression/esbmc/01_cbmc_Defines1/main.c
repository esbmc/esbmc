int main() {
#ifdef NEW_DEFINE
  // Ok!
#else
  // define is missing
  assert(0);
#endif
}
