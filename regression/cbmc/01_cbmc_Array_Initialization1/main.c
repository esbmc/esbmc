// do both orderings

extern const char *abc1[];
const char *abc1[] = {"Hallo", "Welt"};

const char *abc2[] = {"Hallo", "Welt"};
extern const char *abc2[];

int main()
{
  // both must be complete
  sizeof(abc1);
  sizeof(abc2);
}

// modifiers

static const char * const a1[] = { "abc" };
static const char * const a2[] = { "abc", "" };

