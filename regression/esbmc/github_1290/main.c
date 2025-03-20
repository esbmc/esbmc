//FormAI DATASET v1.0 Category: Alien Language Translator ; Style: synchronous
#include <stdio.h>
#include <string.h>

void translate(char* str) {
    // Define the rules of the alien language
    int len = strlen(str);
    char translated_str[len+1];
    int vowel_check[len+1];
    memset(vowel_check, 0, sizeof(vowel_check));
    memset(translated_str, 0, sizeof(translated_str));

    for(int i=0; i<len; ++i) {
        if(str[i] == 'a' || str[i] == 'e' || str[i] == 'i' || str[i] == 'o' || str[i] == 'u') {
            vowel_check[i] = 1;
        }
    }

    int pos = 0;
    for(int i=0; i<len; ++i) {
        if(vowel_check[i]) {
            if(i+3 < len && !vowel_check[i+1] && !vowel_check[i+2] && !vowel_check[i+3]) {
                strncpy(translated_str+pos, str+i+1, 2);
                translated_str[pos+2] = str[i];
                pos += 3;
                i += 3;
            } else {
                translated_str[pos++] = str[i];
            }
        } else {
            translated_str[pos++] = str[i];
        }
    }

    printf("Translated Alien Language: %s\n", translated_str);
}

int main() {
    char str[100];
    printf("Enter the Alien's language to translate: ");
    scanf("%[^\n]%*c", str);

    translate(str);
    return 0;
}