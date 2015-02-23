class CreateAndDestroy {

public:
   CreateAndDestroy( int, char * );  // constructor
//   ~CreateAndDestroy();              // destructor
}; 

//static CreateAndDestroy first( 1, "(global before main)" );

int main()
{

   static CreateAndDestroy second( 2, "(local automatic in main)" );
   return 0;
}
