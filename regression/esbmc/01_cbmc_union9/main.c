// Test union assignment (const propagates, mostly)

typedef union {
  unsigned int a;
  short b[2];
} facet;

facet bees = { 0x12345678 };
facet desk;

int
main()
{
  desk = bees;
  assert(desk.b[0] == 0x5678);
  assert(desk.b[1] == 0x1234);
  return 0;
}
