#include <stdio.h>
#include <assert.h>

int main(int argc, char *argv[])
{
    // Test case 1: Basic functionality - extra argument should remain unchanged
    {
        int m, n = 42;
        fscanf(stdin, "%10d", &m, &n);  // Only %10d, so only &m processed
        assert(n == 42);  // n should remain unchanged
    }
    
    // Test case 2: Multiple format specifiers - all should be processed
    {
        int a = 100, b = 200, c = 300;
        fscanf(stdin, "%10d %10d", &a, &b, &c);  // Two %10d, so &a and &b processed
        assert(c == 300);  // c should remain unchanged
    }
    
    // Test case 3: String format specifier with width
    {
        char str[20] = "hello";
        int num = 42;
        int extra = 999;
        
        fscanf(stdin, "%19s %10d", str, &num, &extra);  // 2 specifiers
        assert(extra == 999);  // extra should remain unchanged
    }
    
    // Test case 4: Length modifiers
    {
        long long_val = 12345L;
        short short_val = 99;
        int regular = 555;
        
        fscanf(stdin, "%15ld %5hd", &long_val, &short_val, &regular);  // 2 specifiers
        assert(regular == 555);  // regular should remain unchanged
    }
    
    // Test case 5: Test scanf variant
    {
        int m = 50, n = 60;
        scanf("%10d", &m, &n);  // Only one %10d
        assert(n == 60);  // n should remain unchanged
    }
    
    // Test case 6: Test sscanf variant  
    {
        int a = 10, b = 20, c = 30;
        sscanf("123", "%10d", &a, &b, &c);  // Only one %10d
        assert(b == 20);  // b should remain unchanged
        assert(c == 30);  // c should remain unchanged
    }

    return 0;
}
