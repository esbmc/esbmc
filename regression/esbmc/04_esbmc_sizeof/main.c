struct face {
  char bees;
  short beards;
  int noses;
};

struct loltrails {
  int ponies;
  char doom;
};

union deathcakes {
  struct loltrails wat;
  int beans;
  char policy;
};

int
main()
{
  assert(sizeof(char) == 1);
  assert(sizeof(short) == 2);
  assert(sizeof(int) == 4);
  assert(sizeof(long) == 4);

  char carr[123];
  assert(sizeof(carr) == 123);

  short sarr[123];
  assert(sizeof(sarr) == 246);

  int iarr[123];
  assert(sizeof(iarr) == 492);

  // Test for padding
  assert(sizeof(struct face) == 12);

  // Test that trailing bytes are added to sizeof to align.
  assert(sizeof(struct loltrails) == 8);

  // Array size allocations
  struct face fa[123];
  assert(sizeof(fa) == 1476);

  // Arrays also need to do trailing bytes
  struct loltrails fb[123];
  assert(sizeof(fb) == 984);

  // Unions max size is the size of the largest member.
  assert(sizeof(union deathcakes) == 8);
  return 0;
}
