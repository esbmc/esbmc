#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <uchar.h>
#include <wchar.h>
#include <locale.h>
#include <assert.h>
#include <string.h>

int main ()
{
    char s1[] = "a猫🍌"; // or "a\u732B\U0001F34C"
    char s2[] = u8"a猫🍌";
    char16_t s3[] = u"a猫🍌";
    char32_t s4[] = U"a猫🍌";
    wchar_t s5[] = L"a猫🍌";
 
    setlocale(LC_ALL, "en_US.utf8");
    printf("  \"%s\" is a char[%zu] holding { ", s1, sizeof s1 / sizeof *s1);
    for(size_t n = 0; n < sizeof s1 / sizeof *s1; ++n) 
        printf("%#x ", +(unsigned char)s1[n]); puts(" }");
    assert(strlen(s1) == 8); // without null terminator
    assert(sizeof(s1) == 9); // with null terminator

    printf("u8\"%s\" is a char[%zu] holding { ", s2, sizeof s2 / sizeof *s2);
    for(size_t n = 0; n < sizeof s2 / sizeof *s2; ++n) 
       printf("%#x ", +(unsigned char)s2[n]); puts(" }");
    assert(strlen(s2) == 8); // without null terminator
    assert(sizeof(s2) == 9); // with null terminator

    printf(" u\"a猫🍌\" is a char16_t[%zu] holding { ", sizeof s3 / sizeof *s3);
    for(size_t n = 0; n < sizeof s3 / sizeof *s3; ++n) 
       printf("%#x ", s3[n]); puts(" }");
    assert((sizeof s3 / sizeof *s3) == 5); // without null terminator

    printf(" U\"a猫🍌\" is a char32_t[%zu] holding { ", sizeof s4 / sizeof *s4);
    for(size_t n = 0; n < sizeof s4 / sizeof *s4; ++n) 
       printf("%#x ", s4[n]); puts(" }");
    assert((sizeof s4 / sizeof *s4) == 4); // without null terminator

    printf(" L\"%ls\" is a wchar_t[%zu] holding { ", s5, sizeof s5 / sizeof *s5);
    for(size_t n = 0; n < sizeof s5 / sizeof *s5; ++n) 
       printf("%#x ", s5[n]); puts(" }");
    assert(wcslen(s5) == 3); // without null terminator
}
