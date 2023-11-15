#include<cassert>
   
class Enclosing {      
   private:   
       int x;
     
   class Nested {
      int y;   
      void func(Enclosing *e) {
        assert(1);  
      }       
   }; 
   public:
    void func()
    {
      assert(0);
    }
}; 
  
int main()
{     
  return 0;
}