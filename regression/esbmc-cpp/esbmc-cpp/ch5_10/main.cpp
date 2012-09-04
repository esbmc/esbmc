// Fig. 5.13: fig05_13.cpp
// Attempting to modify a constant pointer to
// non-constant data.

int main()
{
   int x, y;

   // ptr is a constant pointer to an integer that can
   // be modified through ptr, but ptr always points to the 
   // same memory location.
   int * const ptr = &x; 

   *ptr = 7;  // allowed: *ptr is not const
   ptr = &y;  // error: ptr is const; cannot assign new address

   return 0;  // indicates successful termination

} // end main

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
