#include <cstring>
#include <cstdio>

class PointException
{
public:
  PointException ()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class PolyException
{
public:
  PolyException ()
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
  void main_reshape ()
  {
  }
  void reshape ()
  {
  }
};

class Point:public Shape
{
public:
  int x, y;
  Point ()
  {
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
  void reshapeP ()
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
  void reshapeT ()
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
  void reshapeR ()
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
  void reshapeS ()
  {
  }
};


int
main ()
{
  int i, j, k;

  try
  {
    Rectangle *r = new Rectangle (10, 3);

    for (int i = 0; i < 2; ++i)
    {
      Square *s = new Square (3);

      try
      {

        for (int j = 0; j < 2; ++j)
        {
          Point *p = new Point (4, 5);

          try
          {

            for (k = 0; k < 2; ++k)
            {
              Triangle *t = new Triangle (i, j, k);

              (*t).reshapeT ();
              if (t->a < t->b)
                throw PolyException ();
              else
                if (t->a == t->b)
                  break;

              delete t;
            }
          }
          catch (PolyException & ep)
          {
            printf ("Caught a polygon int\n");
          }

          (*p).reshapeP ();
          if (p->x < p->y)
            throw PointException ();
          else
            if (p->x == p->y)
              continue;

          delete p;
        }
      }
      catch (PointException & ep)
      {
        printf ("Caught a point exception\n");
      }

      (*s).reshapeS ();

      delete s;
    }

    (*r).reshapeR ();
    delete r;
  }
  catch (PointException & epp)
  {
    printf ("Caught a point exception\n");
  }
  catch (...)
  {
    printf ("Unknown exception caught\n");
  }
  printf ("Exit!\n");
  return 0;
}
