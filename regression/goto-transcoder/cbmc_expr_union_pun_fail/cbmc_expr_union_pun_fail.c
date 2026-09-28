union U { unsigned int i; unsigned char b[4]; };
int main() {
  union U u;
  u.i = 0x01020304u;
  __CPROVER_assert(u.b[0] == 0x04 || u.b[0] == 0x01, "little- or big-endian first byte");
  __CPROVER_assert(u.b[0] == 0x01, "wrong endianness");
  return 0;
}
