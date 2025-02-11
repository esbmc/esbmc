//treat with a base class object like a derived class object
//it can generate a compilation error

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
