#include <cstring>
#include <cstdio>
#include <cstdlib>

class ReshapeException
{
public:
  ReshapeException ()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class TriangleReshapeException
{
public:
  TriangleReshapeException ()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class SquareReshapeException
{
public:
  SquareReshapeException ()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class RectangleReshapeException
{
public:
  RectangleReshapeException ()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class PointReshapeException
{
public:
  PointReshapeException ()
  {
    goto ERROR;
    ERROR:
    ;
  }
};

int
hrandom ()
{
  return random() % 2;
}

class Shape
{
public:
  Shape ()
  {
  }
  ~Shape ()
  {
  }
  virtual void reshape ()
  {
    if (hrandom())
      throw ReshapeException ();
  }
};

class Point:public Shape
{
public:
  int x, y;
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
    if (hrandom ())
      throw PointReshapeException ();
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
    if (hrandom())
      throw TriangleReshapeException ();
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
    if (hrandom())
      throw RectangleReshapeException ();
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
    if (hrandom()) // this exception will not be caught.
      throw SquareReshapeException ();
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
reShape (Shape * shapes[])
{
  try
  {
    shapes[0]->reshape ();
    shapes[1]->reshape ();
    shapes[2]->reshape ();
    shapes[3]->reshape ();
  }
  catch (PointReshapeException & s)
  {
    printf ("Caught a Point reshape exception\n");
  }
  catch (RectangleReshapeException & s)
  {
    printf ("Caught a Rectangle reshape exception\n");
  }
  catch (TriangleReshapeException & r)
  {
    printf ("Caught a Point reshape exception\n");
  }
  catch (ReshapeException & s)
  {
    printf ("Caught a reshape exception\n");
  }
}

void
deleteShapes (Shape * shapes[])
{
  delete shapes[0];
  delete shapes[1];
  delete shapes[2];
  delete shapes[3];
}

int
main ()
{
  Shape *shapes[4] = { NULL, NULL, NULL, NULL };
  try
  {
    createShapes (shapes);
    reShape (shapes);
    deleteShapes (shapes);
  }
  catch (...)
  {
    printf ("Unknown exception caught\n");
  }
  return 0;
}
