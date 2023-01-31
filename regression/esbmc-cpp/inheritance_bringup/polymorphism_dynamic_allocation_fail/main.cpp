// dynamic allocation and polymorphism
#include <cassert>

class CPolygon {
  protected:
    int width, height;
  public:
    void set_values (int a, int b)
      { width=a; height=b; }
    virtual int area (void) =0;
    int getArea (void)
      { return this->area(); }
  };

class CRectangle: public CPolygon {
  public:
    int area (void)
      { return (width * height); }
  };

class CTriangle: public CPolygon {
  public:
    int area (void)
      { return (width * height / 2); }
  };

int main () {
  CPolygon * ppoly1 = new CRectangle;
  CPolygon * ppoly2 = new CTriangle;
  ppoly1->set_values (4,5);
  ppoly2->set_values (4,5);

  assert(ppoly1->getArea() == 20);
  assert(ppoly2->getArea() == 20); // FAIL, should be 10

  delete ppoly1;
  delete ppoly2;
  return 0;
}
