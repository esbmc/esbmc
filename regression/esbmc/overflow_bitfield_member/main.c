int nondet_int(void);

struct
{
  int a : 3;
} b = {3};

int main()
{
  // --overflow-check does not check arithmetic on a bitfield member, so this
  // reports SUCCESSFUL although 3 + nondet overflows. The same statement on a
  // plain member is checked; see overflow_plain_member.
  int a = nondet_int();
  b.a += a;
}
