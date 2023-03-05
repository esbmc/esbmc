/*
 * dynamic cast test: cast derived class ptr to base class ptr, late binding
 */
#include <cassert>

class CPolygon {
  protected:
    int width, height;
  public:
  CPolygon(){}
  void set_values (int a, int b)
    { width=a; height=b; }
  virtual int area (void) =0;
};

class CRectangle: public CPolygon {
  public:
  CRectangle(int w, int h)
  {
    width = w;
    height = h;
  }
  int area (void)
    { return (width * height); }
};

int main () {

  CRectangle* polygons = new CRectangle(20,30);

  // cast derived class ptr to base class ptr
  CPolygon* rec = dynamic_cast <CPolygon *> (polygons);
  if (rec != 0)
  {
    rec->set_values(10, 10);
    assert(rec->area() == 100);
  }

   return 0;
}
