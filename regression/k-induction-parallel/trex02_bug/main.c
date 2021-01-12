_Bool nondet_bool();
int nondet_int();

//x is an input variable
int x;

void foo() {
  x--;
}

int main() {
  x=nondet_int();
  while (x > 0) {
    _Bool c = nondet_bool();
    if(c) foo();
    else foo();
  }
  assert(x==0);
}



