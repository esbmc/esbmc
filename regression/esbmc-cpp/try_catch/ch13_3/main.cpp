// Fig. 13.3: fig13_03.cpp
// Demonstrating stack unwinding.
#include <iostream>
using std::cout;
using std::endl;

#include <stdexcept>
using std::runtime_error;

// function3 throws run-time error
void function3() throw ( runtime_error )
{
  throw runtime_error( "runtime_error in function3" ); // fourth
}

// function2 invokes function3
void function2() throw ( runtime_error )
{
  function3(); // third
}

// function1 invokes function2
void function1() throw ( runtime_error )
{
  function2(); // second
}

// demonstrate stack unwinding
int main()
{
  // invoke function1
  try {
    function1(); // first
  } // end try

  // handle run-time error
  catch ( runtime_error &error ) // fifth
  {
    cout << "Exception occurred: " << error.what() << endl;
  } // end catch

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
