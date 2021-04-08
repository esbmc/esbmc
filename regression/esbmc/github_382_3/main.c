void foo(unsigned int *p){
    *p = 5; // p is pointing to address 0X06
}

int main() {
   int x = 6;
   unsigned int *p = (unsigned int *)x;
   foo(p);
   return 0; 
}

