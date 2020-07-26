#include <cstdio>
#include <cstdlib>
class InvalidRectangleException
{
public:
  InvalidRectangleException()
  {
    goto ERROR;
    ERROR:
    ;
  }

};
class InvalidDoorException
{
public:
  InvalidDoorException()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class InvalidRoomException
{
public:
  InvalidRoomException()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class IndexOutOfBoundException
{
public:
  IndexOutOfBoundException()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class NonEmptyHouseException
{
public:
  NonEmptyHouseException()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class HouseException
{
public:
  HouseException()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class AlreadyOccupiedException
{
public:
  AlreadyOccupiedException()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class NotOccupiedException
{
public:
  NotOccupiedException()
  {
    goto ERROR;
    ERROR:
    ;
  }
};

class Rectangle
{
public:
  int x, y;
  int *cells;
  Rectangle ()
  {
    printf ("Constructed Rectange\n");
    x = -1;
    y = -1;
    cells = NULL;
  }
  Rectangle (int px, int py)
  {
    printf ("Constructed Rectangle\n");
    x = px;
    y = py;
    cells = new int[px * py];
  }

  ~Rectangle ()
  {
    if (x < 0 || y < 0)
      throw InvalidRectangleException ();
    printf ("Destroyed Rectange\n");
  }
};


class Door
{
public:
  Rectangle * r;
  Door ()
  {
    printf ("Constructed Door\n");
    r = NULL;
  }
  Door (Rectangle * pr)
  {
    printf ("Constructed Door\n");
    r = pr;
  }
  ~Door ()
  {
    if (!r)
      throw InvalidDoorException ();
    delete r;
    printf ("Destroyed Door\n");
  }
};

class Room
{
public:
  Door * r;
  Room ()
  {
    printf ("Constructed Room\n");
    r = NULL;
  }
  Room (Door * pr)
  {
    printf ("Constructed Room\n");
    r = pr;
  }
  ~Room ()
  {
    if (!r)
      throw InvalidRoomException ();
    delete r;
    printf ("Destroyed Room\n");
  }
};

class House
{
public:
  int numRooms;
  Room **rooms;
  bool isOccupied;
  House ()
  {
    printf ("Constructed House\n");
    rooms = NULL;
    numRooms = 0;
    isOccupied = false;
  }

  House (int numR)
  {
    printf ("Constructed House\n");
    numRooms = numR;
    rooms = new Room *[numRooms];
    isOccupied = false;
  }

  void create (int numR)
  {
    if (numRooms != 0)
      throw NonEmptyHouseException ();
    numRooms = numR;
    rooms = new Room *[numRooms];
    isOccupied = false;
    printf ("Constructed House\n");
  }

  void assignRoom (int index, Room * r)
  {
    if (index > numRooms || index < 0)
      throw IndexOutOfBoundException ();
    rooms[index] = r;
  }

  void occupy ()
  {
    if (isOccupied)
      throw AlreadyOccupiedException ();
    isOccupied = true;
  }

  void leave ()
  {
    if (!isOccupied)
      throw NotOccupiedException ();
    isOccupied = false;
  }

  ~House ()
  {
    if (!rooms)
      throw HouseException ();
    for (int i = 0; i < numRooms; ++i)
    {
      Room *r = rooms[i];
      delete r;
    }
    printf ("Destroyed House\n");
  }
};

int
createSmallHouse (House & h)
{
  Rectangle *r1 = NULL, *r2 = NULL, *r3 = NULL;

  r1 = new Rectangle ();
  r1->x = 4;
  r1->y = 5;
  r2 = new Rectangle ();
  r2->x = 8;
  r2->y = 9;
  r3 = new Rectangle ();
  r3->x = 10;
  r3->y = 11;

  Door *d1 = new Door (r1);
  Door *d2 = new Door (r2);
  Door *d3 = new Door (r3);

  Room *ro1 = new Room (d1);
  Room *ro2 = new Room (d2);
  Room *ro3 = new Room (d3);

  h.create (3);
  h.assignRoom (0, ro1);
  h.assignRoom (1, ro2);
  h.assignRoom (2, ro3);
}

int
createBigHouse (House & h)
{
  Rectangle *r1 = NULL, *r2 = NULL, *r3 = NULL;

  r1 = new Rectangle ();
  r1->x = 49;
  r1->y = 53;
  r2 = new Rectangle ();
  r2->x = 82;
  r2->y = 94;

  Door *d1 = new Door (r1);
  Door *d2 = new Door (r2);
  Door *d3 = new Door (r3);

  Room *ro1 = new Room (d1);
  Room *ro2 = new Room (d2);
  Room *ro3 = new Room (d3);

  h.create (3);
  h.assignRoom (0, ro1);
  h.assignRoom (1, ro2);
  h.assignRoom (2, ro3);
}

int
hrandom ()
{
  return random () % 2;
}

int
main ()
{
  int numHouses = 2;
  try
  {
    if (numHouses == 2)
    {
      House h1;
      {
        if (hrandom ())
        {
          House h2;
          createBigHouse (h1);
          createSmallHouse (h2);
          h2.occupy ();
          // The following stmt will  throw an exception at which
          // point destructor of h2 is called, which in turn will
          // throw an exception which should cause termination
          h2.occupy ();
        }
        else
        {
          House h2;
          createBigHouse (h1);
          createSmallHouse (h2);
          h2.occupy ();
          h1.occupy ();
          h2.leave ();
        }
        h1.leave ();
      }
    }
    else
      printf ("More than 2 houses not supported\n");
  }
  catch (...)
  {
    printf ("Exception Caught\n");
  }
}
