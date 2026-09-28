// One assignment per member of a wide struct -- the shape that made
// migrate_type_back() rebuild the legacy struct type per use (#7571).
struct wide {
  int f0;
  int f1;
  int f2;
  int f3;
  int f4;
  int f5;
  int f6;
  int f7;
  int f8;
  int f9;
  int f10;
  int f11;
  int f12;
  int f13;
  int f14;
  int f15;
  int f16;
  int f17;
  int f18;
  int f19;
  int f20;
  int f21;
  int f22;
  int f23;
  int f24;
  int f25;
  int f26;
  int f27;
  int f28;
  int f29;
  int f30;
  int f31;
};

struct wide g;

int main(void)
{
  g.f0 = 0;
  g.f1 = 1;
  g.f2 = 2;
  g.f3 = 3;
  g.f4 = 4;
  g.f5 = 5;
  g.f6 = 6;
  g.f7 = 7;
  g.f8 = 8;
  g.f9 = 9;
  g.f10 = 10;
  g.f11 = 11;
  g.f12 = 12;
  g.f13 = 13;
  g.f14 = 14;
  g.f15 = 15;
  g.f16 = 16;
  g.f17 = 17;
  g.f18 = 18;
  g.f19 = 19;
  g.f20 = 20;
  g.f21 = 21;
  g.f22 = 22;
  g.f23 = 23;
  g.f24 = 24;
  g.f25 = 25;
  g.f26 = 26;
  g.f27 = 27;
  g.f28 = 28;
  g.f29 = 29;
  g.f30 = 30;
  g.f31 = 31;

  __ESBMC_assert(g.f0 == 0, "first member");
  __ESBMC_assert(g.f16 == 16, "middle member");
  __ESBMC_assert(g.f31 == 31, "last member");
  return 0;
}
