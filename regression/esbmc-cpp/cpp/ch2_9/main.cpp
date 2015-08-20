// Fig. 2.11: fig02_11.cpp
// Analysis of examination results.
#include <iostream>

using std::cout;
using std::cin;
using std::endl;

// function main begins program execution
int main()
{
   // initialize variables in declarations
   int passes = 0;           // number of passes
   int failures = 0;         // number of failures
   int studentCounter = 1;   // student counter
   int result;               // one exam result

   // process 10 students using counter-controlled loop
   while ( studentCounter <= 10 ) {

      // prompt user for input and obtain value from user
      cout << "Enter result (1 = pass, 2 = fail): ";
      cin >> result;

      // if result 1, increment passes; if/else nested in while
      if ( result == 1 )        // if/else nested in while
         passes = passes + 1;

      else  // if result not 1, increment failures
         failures = failures + 1;

      // increment studentCounter so loop eventually terminates
      studentCounter = studentCounter + 1; 
      
   } // end while 

   // termination phase; display number of passes and failures
   cout << "Passed " << passes << endl;  
   cout << "Failed " << failures << endl;

   // if more than eight students passed, print "raise tuition"
   if ( passes > 8 )
      cout << "Raise tuition " << endl; 

   return 0;   // successful termination

} // end function main



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
