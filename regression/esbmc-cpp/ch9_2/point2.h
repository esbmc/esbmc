// Fig. 9.12: point2.h
// Point2 class definition represents an x-y coordinate pair.
#ifndef POINT2_H
#define POINT2_H

class Point2 {

public:
   Point2( int = 0, int = 0 ); // default constructor

   void setX( int );    // set x in coordinate pair
   int getX() const;    // return x from coordinate pair
   
   void setY( int );    // set y in coordinate pair
   int getY() const;    // return y from coordinate pair
   
   void print() const;  // output Point2 object

protected: 
   int x;  // x part of coordinate pair
   int y;  // y part of coordinate pair

}; // end class Point2

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