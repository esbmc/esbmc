int main( )
{
int a;
int b;
int c;
a = nondet_int();
__CPROVER_input("a",a);
b = 1;
c = nondet_int(); 
__CPROVER_input("c",c);
if (a > 5)
{
printf("Condition 1 is true");
if (b == 0 && c > 90)  
{
printf("Conditions 2 & 3 are true");
}
else
{
printf("Condition 2 is false");
}
}
else {
printf("Condition 1 is false");
}
}
