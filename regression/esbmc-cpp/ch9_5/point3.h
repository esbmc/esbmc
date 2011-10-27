// Fig. 9.17: point3.h
// Point3 class definition represents an x-y coordinate pair.
#ifndef POINT3_H
#define POINT3_H

class Point3 {

public:
   Point3( int = 0, int = 0 ); // default constructor

   void setX( int );    // set x in coordinate pair
   int getX() const;    // return x from coordinate pair
   
   void setY( int );    // set y in coordinate pair
   int getY() const;    // return y from coordinate pair
   
   void print() const;  // output Point3 object

private: 
   int x;  // x part of coordinate pair
   int y;  // y part of coordinate pair

}; // end class Point3

#endif

/**************************************************************************
 * (C) Copyright 1992-2003 by Deitel & Associates, Inc. and Prentice      *
 * Hall. All Rights Reserved.                                             *
 *                                                                        *
 * DISCLAIMER: The authors and publisher of this book have used their     *
 * best efforts in preparing the book. These efforts include the          *
 * development, research, and testing of the theories and programs        *
 * to determine their effectiveness. The authors and publisher make       *
 * no warranty of any kind, expressed or implied, with regard to these    *
 * programs or to the documentation contained in these books. The authors *
 * and publisher shall not be liable in any event for incidental or       *
 * consequential damages in connection with, or arising out of, the       *
 * furnishing, performance, or use of these programs.                     *
 *************************************************************************/