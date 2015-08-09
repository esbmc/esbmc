// Check initialization of arrays decomposes correctly

union {
  unsigned int a;
  short b[2];
} face = { 0x12345678 };

int
main()
{
  assert(face.b[0] == 0x5678);
  assert(face.b[1] == 0x1234);
  return 0;
}
