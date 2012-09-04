// Fig. 3.4: fig03_04.cpp
// Finding the maximum of three floating-point numbers.
#include <iostream>

using std::cout;
using std::cin;
using std::endl;

double maximum( double, double, double ); // function prototype

int main()
{
   double number1;
   double number2;
   double number3;

   cout << "Enter three floating-point numbers: "<< endl;
   cin >> number1 >> number2 >> number3;

   // number1, number2 and number3 are arguments to 
   // the maximum function call
   cout << "Maximum is: " 
        << maximum( number1, number2, number3 ) << endl;

   return 0;  // indicates successful termination

} // end main

// function maximum definition;
// x, y and z are parameters
double maximum( double x, double y, double z )
{
   double max = x;   // assume x is largest

   if ( y > max )    // if y is larger,
      max = y;       // assign y to max

   if ( z > max )    // if z is larger,
      max = z;       // assign z to max
  
   return max;       // max is largest value

} // end function maximum



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
