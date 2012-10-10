// Fig. 13.2: fig13_02.cpp
// Demonstrating exception rethrowing.
#include <iostream>
#include <cassert>
using std::cout;
using std::endl;

#include <exception>
using std::exception;

// throw, catch and rethrow exception
void throwException()
{
  // throw exception and catch it immediately
  try {
    cout << "  Function throwException throws an exception\n";
    throw exception();  // generate exception
  } // end try

  // handle exception
  catch ( exception &caughtException ) {
    cout << "  Exception handled in function throwException"
        << "\n  Function throwException rethrows exception";
    assert(0);
    throw;  // rethrow exception for further processing
  } // end catch
  cout << "This also should not print\n";
}  // end function throwException

int main()
{
  // throw exception
  try {
    cout << "\nmain invokes function throwException\n";
    throwException();
    cout << "This should not print\n";
  } // end try
  // handle exception
  catch ( exception &caughtException ) {
    cout << "\n\nException handled in main\n";
  } // end catch

  cout << "Program control continues after catch in main\n";
  return 0;
}  // end main


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
