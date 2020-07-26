#include <cstdio>

class InvalidSideException
{
public:
  InvalidSideException ()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class TriangleInequalityException
{
public:
  TriangleInequalityException ()
  {
    goto ERROR;
    ERROR:
    ;
  }
};

class Shape
{
public:
  Shape ()
  {
  }
  ~Shape ()
  {
  }
};

class Point
{
public:
  int id;
  Point ()
  {
  }
  ~Point ()
  {
  }
};

// Triangle is derived from Shape.
class Triangle:public Shape
{
public:
  int base;
  int left;
  int right;
  Point *points;
  int np;

  Triangle (int b, int l, int r)
  {
    printf ("Triangle constructor called\n");
    base = b;
    left = l;
    right = r;

    if (base <= 0 || left <= 0 || right <= 0)
      throw InvalidSideException ();

    int s = (base + left + right) / 2;
    np = base;
    points = new Point[base];

    if ((base + left) <= right || (base + right) <= left
        || (left + right) <= base)
      throw TriangleInequalityException ();

    printf ("Triangle constructor done\n");
  }

  int printTriangle ()
  {

  }

  ~Triangle ()
  {
    delete[]points;
  }
};


int
main ()
{

  try
  {
    Triangle t1 (4, 2, 4);
    Triangle t2 (6, 8, 4);
    for (int i = 3; i >= 0; --i)
    {
      try
      {
        // Destructors for throwing constructors should not be called
        Triangle t3 (i, 2 * i, 3 * i + 2);
        t3.printTriangle ();
      }
      catch (TriangleInequalityException & ex1)
      {
        Point x;
        printf ("Loop TriangleInequalityException caught!\n");
      }
      catch (InvalidSideException & s)
      {
        Point x;
        printf ("Loop InvalidSideException caught!\n");
      }
    }
  }
  catch (TriangleInequalityException & ex1)
  {
    Point x;
    printf ("TriangleInequalityException caught!\n");
  }
  catch (InvalidSideException & s)
  {
    Point x;
    printf ("InvalidSideException caught!\n");
  }

}
