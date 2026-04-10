
extern int __VERIFIER_nondet_int();

int classify(int x) {
    if (x > 0) {
        return 1;
    } else if (x < 0) {
        return -1;
    } else {
        return 0;
    }
}

int nested_check(int a, int b) {
    if (a > 0) {
        if (b > 0) {
            return 1;
        } else {
            return 2;
        }
    } else {
        if (b > 0) {
            return 3;
        } else {
            return 4;
        }
    }
}

int safe_div(int x, int y) {
    if (y == 0) {
        return 0;  
    }
    return x / y;
}

int main() {
    int a = __VERIFIER_nondet_int();
    int b = __VERIFIER_nondet_int();
    
    int r1 = classify(a);
    int r2 = nested_check(a, b);
    int r3 = safe_div(a, b);
    
    return 0;
}