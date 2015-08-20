//treat with a base class object like a derived class object
//it can generate a compilation error
#include <iostream>
using namespace std;

class Shape{
  public:
    virtual int area (void);
    virtual void printArea(void);

};

class CPolygon {
  protected:
    int width, height;
  public:
    void set_values (int a, int b)
      { width=a; height=b; }
    virtual int area (void) =0;
    void printarea (void)
      { cout << this->area() << endl; }
  };

class CRectangle: public CPolygon {
  public:
    int area (void)
      { return (width * height); }
    int randomVariable;
  };

class CTriangle: public CPolygon {
  public:
    long area (void)
      { return (width * height / 2); }
    int randomVariable2;
  };

int main () {
  CPolygon *trin = new CTriangle;
  trin.set_values(10,10);
  cout << trin.area() << endl; 
  
  return 0;
}
