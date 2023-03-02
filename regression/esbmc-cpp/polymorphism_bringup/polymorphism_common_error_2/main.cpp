//treat with a base class object like a derived class object
//it can generate a compilation error

class CPolygon {
protected:
  int width, height;
public:
  void set_values (int a, int b)
    { width=a; height=b; }
  virtual int area (void) =0;
};

class CRectangle: public CPolygon {
public:
  int area (void)
    { return (width * height); }
  int randomVariable;
};

class CTriangle: public CPolygon {
public:
  int area (void)
    { return (width * height / 2); }
  int randomVariable2;
};

int main () {
  CPolygon *polygon = new CPolygon;
  polygon.set_values(10, 10);

  return 0;
}
