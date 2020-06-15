#include <cstring>
#include <cstdio>
#include <new>

class Shape
{
public:
  Shape ()
  {
    printf ("consr of Shape called \n");
  }
  ~Shape ()
  {
  }
  void main_reshape ()
  {
  }
  virtual void reshape ()
  {
  }
};

class Point:public Shape
{
public:
  int x, y;
  Point ()
  {
    printf ("consr of Point called \n");
    x = y = -1;
  }
  Point (int px, int py)
  {
    x = px;
    y = py;
  }
  ~Point ()
  {
  }
  virtual void reshape ()
  {
  }
  void reshapePoint ()
  {
  }
};

// Triangle is derived from Shape.
class Triangle:public Shape
{
public:
  int a, b, c;
  Triangle (int s1, int s2, int s3)
  {
    a = s1;
    b = s2;
    c = s3;
  }
  virtual void reshape ()
  {
  }
};


class Rectangle:public Shape
{
public:
  int l, w;
  Rectangle (int len, int width)
  {
    l = len;
    w = width;
  }
  virtual void reshape ()
  {
  }
};

class Square:public Shape
{
public:
  int e;
  Square (int s)
  {
    e = s;
  }
  virtual void reshape ()
  {
  }
};

void
createShapes (Shape * shapes[])
{
  shapes[0] = new Point (1, 2);
  shapes[1] = new Triangle (2, 3, 4);
  shapes[2] = new Rectangle (3, 4);
  shapes[3] = new Square (4);
}

void
reShapePoints (Point * shape, int n)
{
  for (int i = 0; i < n; ++i)
  {
    try
    {
      Point *tmp1 = new Point (8, 10);
      Point *tmp2 = new Point (28, 2);

      Point & p = shape[i];
      p.x = tmp1->x + tmp2->x;
      p.y = tmp1->y + tmp2->y;

      delete tmp1;
      delete tmp2;
    }
    catch (std::bad_alloc & ba)
    {
      printf ("Caught bad alloc\n");
      goto ERROR;
      ERROR:
      ;
    }
  }
}

int
main ()
{
  printf ("from here \n");
  Point *p = new Point[2];
  printf ("end here \n");

  try
  {
    reShapePoints (p, 2);
  }
  catch (...)
  {
    printf ("Unknown exception caught\n");
    goto ERROR;
    ERROR:
    ;
  }
  delete[] p;
  return 0;
}
