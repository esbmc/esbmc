//common error: after setting a base class pointer to an derived class object, attempt to reference exclusive members of derived class with base class pointer
//it generates a compilation error

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
  CRectangle *rect = new CPolygon;
  rect.randomVariable = 15;
  CTriangle *trin = new CPolygon;
  trin.randomVariable2 = 20;

  return 0;
}
