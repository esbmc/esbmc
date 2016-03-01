
// A normal function with an int parameter
// and void return type
void fun(int a)
{
    printf("Value of a is %d\n", a);
}

void fun1()
{
  printf("No value\n");
}

int fun2()
{
  return 2;
}

int main()
{
    // fun_ptr is a pointer to function fun() 
    void (*fun_ptr)(int) = &fun;
    (*fun_ptr)(10);
     
    void (*fun_ptr1)(int);
    fun_ptr1 = &fun; 
    (*fun_ptr1)(10);

    void (*fun_ptr2)() = &fun1;
    (*fun_ptr2)();
     
    void (*fun_ptr3)();
    fun_ptr3 = &fun1; 
    (*fun_ptr3)();
 
    int x = fun2();
    fun2();

    return 0;
}
