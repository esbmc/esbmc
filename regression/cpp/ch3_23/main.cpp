// Fig. 3.26: fig03_26.cpp
// Name mangling.

// function square for int values
int square( int x ) 
{ 
   return x * x; 
}

// function square for double values
double square( double y ) 
{ 
   return y * y; 
}

// function that receives arguments of types 
// int, float, char and int *
void nothing1( int a, float b, char c, int *d ) 
{ 
   // empty function body
}  

// function that receives arguments of types 
// char, int, float * and double *
char *nothing2( char a, int b, float *c, double *d ) 
{ 
   return 0; 
}

int main()
{
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
