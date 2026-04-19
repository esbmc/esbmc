// Fig. 13.6: fig13_06.cpp
// Demonstrating set_new_handler.
#include <iostream>

using std::cout;
using std::cerr;

#include <new>     // standard operator new and set_new_handler

using std::set_new_handler;

#include <cstdlib> // abort function prototype

void customNewHandler()
{
   cerr << "customNewHandler was called";
   abort();
}

// using set_new_handler to handle failed memory allocation
int main()
{
   double *ptr[ 5 ];

   // specify that customNewHandler should be called on failed 
   // memory allocation
   set_new_handler( customNewHandler );   

   // allocate memory for ptr[ i ]; customNewHandler will be
   // called on failed memory allocation
   for ( int i = 0; i < 5; i++ ) {
      ptr[ i ] = new double[10];

      cout << "Allocated 10 doubles in ptr[ " 
           << i << " ]\n";

   } // end for

   return 0;

}  // end main
