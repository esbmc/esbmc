// `new Shape` is well-formed: Shape's virtual functions are declared but not
// pure, so Shape is not abstract. Their missing definitions are only a
// link-time error (missing vtable, ill-formed but no diagnostic required per
// [basic.def.odr]) and are never reached here, so ESBMC reports VERIFICATION
// SUCCESSFUL. (ESBMC's legacy C++ frontend wrongly reported CONVERSION ERROR.)

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
      {
        // print something here
      }
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
  Shape *sh = new Shape;

  return 0;
}
