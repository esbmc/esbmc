// Fig. 9.18: point3.cpp
// Point3 class member-function definitions.
#include <iostream>  

using std::cout;

#include "point3.h"   // Point3 class definition

// default constructor
Point3::Point3( int xValue, int yValue )
   : x( xValue ), y( yValue )
{
   // empty body 

} // end Point3 constructor

// set x in coordinate pair
void Point3::setX( int xValue )
{
   x = xValue; // no need for validation

} // end function setX

// return x from coordinate pair
int Point3::getX() const
{
   return x;

} // end function getX

// set y in coordinate pair
void Point3::setY( int yValue )
{
   y = yValue; // no need for validation

} // end function setY

// return y from coordinate pair
int Point3::getY() const
{
   return y;

} // end function getY
   
// output Point3 object
void Point3::print() const
{
   cout << '[' << getX() << ", " << getY() << ']';

} // end function print

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