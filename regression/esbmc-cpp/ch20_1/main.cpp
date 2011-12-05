// Fig. 20.2: fig20_02.cpp
// Using variable-length argument lists
#include <iostream>

using std::cout;
using std::endl;
using std::ios;

#include <iomanip>

using std::setw;
using std::setprecision;
using std::setiosflags;
using std::fixed;

#include <cstdarg>

double average( int, ... );

int main()
{
   double double1 = 37.5;
   double double2 = 22.5;
   double double3 = 1.7;
   double double4 = 10.2;

   cout << fixed << setprecision( 1 ) << "double1 = " 
        << double1 << "\ndouble2 = " << double2 << "\ndouble3 = "
        << double3 << "\ndouble4 = " << double4 << endl
        << setprecision( 3 ) 
        << "\nThe average of double1 and double2 is " 
        << average( 2, double1, double2 )
        << "\nThe average of double1, double2, and double3 is " 
        << average( 3, double1, double2, double3 ) 
        << "\nThe average of double1, double2, double3"
        << " and double4 is " 
        << average( 4, double1, double2, double3, double4 ) 
        << endl;

   return 0;

}  // end main

// calculate average
double average( int count, ... )
{
   double total = 0;
   va_list list;  // for storing information needed by va_start
 
   va_start( list, count );

   // process variable length argument list
   for ( int i = 1; i <= count; i++ )
      total += va_arg( list, double );

   // end the va_start
   va_end( list );

   return total / count;

}  // end function average

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
