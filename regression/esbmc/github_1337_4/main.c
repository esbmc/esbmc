   #include <assert.h>

   void loop1()
   {
      for (int i = 0; i < 2; i++)
      {
         assert(1);
      }
   }

   void loop2()
   {
      for (int i = 0; i < 2; i++)
      {
         assert(1);
      }
   }

   int main()

   {
      switch (nondet_int())
      {
      case 1:
         loop1();
      case 2:
         loop2();
      default:
         ;
      }
      return 0;
   }