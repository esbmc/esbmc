/*
 * just dynamic cast
 */
#include <cassert>

class CPolygon {
  protected:
    int width, height;
  public:
  CPolygon(int w, int h)
  {
    width = w;
    height = h;
  }
  void set_values (int a, int b)
    { width=a; height=b; }
  int area (void)
    { return (width * height); }
};

int main () {
  CPolygon* polygons;

  polygons = new CPolygon(20,30);

  CPolygon* rec = dynamic_cast <CPolygon *> (polygons);
  if (rec != 0)
  {
    rec->set_values(10, 10);
    assert(rec->area() == 100);
  }

  return 0;
}
