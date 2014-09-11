//dynamic cast test
//#include <iostream>
//#include <cassert>
//using namespace std;

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

class CTriangle: public CPolygon {
  public:
  CTriangle(int w, int h)
  {
    width = w;
    height = h;
  }
    int area (void)
      { return ((width * height) / 2); }
  };

int main () {
  CPolygon* polygons;

  polygons = new CRectangle(20,30);

  CTriangle* trin = dynamic_cast <CRectangle *> (polygons);
  if (trin != 0)
  {
    trin->set_values(10, 10);
    assert(trin->area() == 300);
  }

  return 0;
}
