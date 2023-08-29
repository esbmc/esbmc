#include <cassert>

class overflow{
  public:
  int x ;
  int y;
  overflow(int _x, int _y)
  { 
    x=_x;
    y=_y;
  }
  void add(int xx, int yy)
  {
    int z = xx+yy;
  } 
};

class overflow2{
  public:
  int x;
  int y;
  overflow2(int _x, int _y)
  { 
    x=_x;
    y=_y;
  }
  void add()
  {
    assert(1);
  } 
};

int main() {
  // overflow x = overflow(nondet_int(),nondet_int());
  return 0;
}
