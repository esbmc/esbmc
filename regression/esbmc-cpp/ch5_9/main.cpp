// Fig. 5.12: fig05_12.cpp
// Attempting to modify data through a
// non-constant pointer to constant data.

void f( const int * );  // prototype

int main()
{
   int y;

   f( &y );   // f attempts illegal modification

   return 0;  // indicates successful termination

} // end main

// xPtr cannot modify the value of the variable 
// to which it points
void f( const int *xPtr )
{
  *xPtr = 100;  // error: cannot modify a const object

} // end function f

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
