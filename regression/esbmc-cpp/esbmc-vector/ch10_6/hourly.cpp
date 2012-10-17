// Fig. 10.28: hourly.cpp
// HourlyEmployee class member-function definitions.
#include <iostream>

using std::cout;

#include "hourly.h"

// constructor for class HourlyEmployee
HourlyEmployee::HourlyEmployee( const string &first, 
   const string &last, const string &socialSecurityNumber,
   double hourlyWage, double hoursWorked )
   : Employee( first, last, socialSecurityNumber )   
{
   setWage( hourlyWage );
   setHours( hoursWorked );

} // end HourlyEmployee constructor

// set hourly worker's wage
void HourlyEmployee::setWage( double wageAmount ) 
{ 
   wage = wageAmount < 0.0 ? 0.0 : wageAmount; 

} // end function setWage

// set hourly worker's hours worked
void HourlyEmployee::setHours( double hoursWorked )
{ 
   hours = ( hoursWorked >= 0.0 && hoursWorked <= 168.0 ) ?
      hoursWorked : 0.0;

} // end function setHours

// return hours worked
double HourlyEmployee::getHours() const
{
   return hours;

} // end function getHours

// return wage
double HourlyEmployee::getWage() const
{
   return wage;

} // end function getWage

// get hourly worker's pay
double HourlyEmployee::earnings() const 
{ 
   if ( hours <= 40 )  // no overtime
      return wage * hours;
   else                // overtime is paid at wage * 1.5
      return 40 * wage + ( hours - 40 ) * wage * 1.5;

} // end function earnings

// print hourly worker's information 
void HourlyEmployee::print() const
{
   cout << "\nhourly employee: ";
   Employee::print();  // code reuse

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