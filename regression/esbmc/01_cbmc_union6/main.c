// Check that we can operate on arrays correctly, now that the underlying
// storage of unions is a byte array

union {
  unsigned int a;
  short b[2];
} face;

int
main()
{
  face.a = 0x12345678;
  assert(face.b[0] == 0x5678);
  assert(face.b[1] == 0x1234);
  return 0;
}
