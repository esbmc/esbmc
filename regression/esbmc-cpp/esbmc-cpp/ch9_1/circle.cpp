// Fig. 9.8: circle.cpp
// Circle class member-function definitions.
#include <iostream>  

using std::cout;

#include "circle.h"   // Circle class definition

// default constructor
Circle::Circle( int xValue, int yValue, double radiusValue )
{
   x = xValue;
   y = yValue;
   setRadius( radiusValue );

} // end Circle constructor

// set x in coordinate pair
void Circle::setX( int xValue )
{
   x = xValue; // no need for validation

} // end function setX

// return x from coordinate pair
int Circle::getX() const
{
   return x;

} // end function getX

// set y in coordinate pair
void Circle::setY( int yValue )
{
   y = yValue; // no need for validation

} // end function setY

// return y from coordinate pair
int Circle::getY() const
{
   return y;

} // end function getY
   
// set radius 
void Circle::setRadius( double radiusValue )
{
   radius = ( radiusValue < 0.0 ? 0.0 : radiusValue );

} // end function setRadius

// return radius 
double Circle::getRadius() const
{
   return radius;

} // end function getRadius

// calculate and return diameter
double Circle::getDiameter() const
{
   return 2 * radius;

} // end function getDiameter

// calculate and return circumference
double Circle::getCircumference() const
{
   return 3.14159 * getDiameter();

} // end function getCircumference

// calculate and return area
double Circle::getArea() const
{
   return 3.14159 * radius * radius;

} // end function getArea

// output Circle object
void Circle::print() const
{
   cout << "Center = [" << x << ", " << y << ']'
        << "; Radius = " << radius;

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