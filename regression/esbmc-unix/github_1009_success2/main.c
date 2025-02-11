#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

int main() {
    char str1[] = "1234567890";
    char str2[] = "9876543210";
    char str3[] = "123456789";
    char str4[] = "-1234567890";
    char str5[] = "+1234567890";
    char str6[] = "  1234567890  ";
    char str7[] = "";
    char str8[] = "    ";
    
    long int num1 = atol(str1);
    long int num2 = atol(str2);
    long int num3 = atol(str3);
    long int num4 = atol(str4);
    long int num5 = atol(str5);
    long int num6 = atol(str6);
    long int num7 = atol(str7);
    long int num8 = atol(str8);
    
    // Test the conversion of positive integers
    assert(num1 == 1234567890L);
    
    // Test the conversion of larger positive integers
    assert(num2 == 9876543210L);
    
    // Test the conversion of positive integers with fewer digits
    assert(num3 == 123456789L);
    
    // Test the conversion of negative integers
    assert(num4 == -1234567890L);
    
    // Test the conversion of positive integers with a plus sign
    assert(num5 == 1234567890L);
    
    // Test the conversion of positive integers with leading and trailing whitespace
    assert(num6 == 1234567890L);
    
    // Test the conversion of empty strings
    assert(num7 == 0L);
    
    // Test the conversion of strings with only whitespace
    assert(num8 == 0L);
    
    return 0;
}

