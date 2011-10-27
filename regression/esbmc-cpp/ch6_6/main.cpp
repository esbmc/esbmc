// Fig. 6.11: fig06_11.cpp
// Demonstrating a utility function.
// Compile this program with salesp.cpp

// include SalesPerson class definition from salesp.h
#include "salesp.h"  

int main()
{
   SalesPerson s;         // create SalesPerson object s
   
   s.getSalesFromUser();  // note simple sequential code; no
   s.printAnnualSales();  // control structures in main

   return 0;

} // end main

/**************************************************************************
 * (C) Copyright 1992-2002 by Deitel & Associates, Inc. and Prentice      *
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
