// Fig. 9.25: point4.h
// Point4 class definition represents an x-y coordinate pair.
#ifndef POINT4_H
#define POINT4_H

class Point4 {

public:
   Point4( int = 0, int = 0 ); // default constructor
   ~Point4();           // destructor

   void setX( int );    // set x in coordinate pair
   int getX() const;    // return x from coordinate pair
   
   void setY( int );    // set y in coordinate pair
   int getY() const;    // return y from coordinate pair
   
   void print() const;  // output Point3 object

private: 
   int x;  // x part of coordinate pair
   int y;  // y part of coordinate pair

}; // end class Point4

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
