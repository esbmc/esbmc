// Fig. 9.20: circle4.cpp
// Circle4 class member-function definitions.
#include <iostream>  

using std::cout;

#include "circle4.h"   // Circle4 class definition

// default constructor
Circle4::Circle4( int xValue, int yValue, double radiusValue )
   : Point3( xValue, yValue )  // call base-class constructor
{
   setRadius( radiusValue );

} // end Circle4 constructor

// set radius 
void Circle4::setRadius( double radiusValue )
{
   radius = ( radiusValue < 0.0 ? 0.0 : radiusValue );

} // end function setRadius

// return radius 
double Circle4::getRadius() const
{
   return radius;

} // end function getRadius

// calculate and return diameter
double Circle4::getDiameter() const
{
   return 2 * getRadius();

} // end function getDiameter

// calculate and return circumference
double Circle4::getCircumference() const
{
   return 3.14159 * getDiameter();

} // end function getCircumference

// calculate and return area
double Circle4::getArea() const
{
   return 3.14159 * getRadius() * getRadius();

} // end function getArea

// output Circle4 object
void Circle4::print() const
{
   cout << "Center = ";
   Point3::print();      // invoke Point3's print function
   cout << "; Radius = " << getRadius();

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