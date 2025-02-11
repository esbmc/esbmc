//FormAI DATASET v0.1 Category: File Encyptor ; Style: Claude Shannon
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void encrypt(char* filename, char* key) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open file: %s\n", filename);
        return;
    }

    char* encrypted_filename = malloc(strlen(filename) + 4);
    strcpy(encrypted_filename, filename);
    strcat(encrypted_filename, ".enc");

    FILE* encrypted_file = fopen(encrypted_filename, "w");
    if (encrypted_file == NULL) {
        printf("Failed to create encrypted file: %s\n", encrypted_filename);
        fclose(file);
        free(encrypted_filename);
        return;
    }

    int key_len = strlen(key);

    char c = fgetc(file);
    int i = 0;
    while (c != EOF) {
        fputc(c ^ key[i % key_len], encrypted_file);
        c = fgetc(file);
        i++;
    }

    fclose(file);
    fclose(encrypted_file);
    free(encrypted_filename);
}

void decrypt(char* filename, char* key) {
    FILE* encrypted_file = fopen(filename, "r");
    if (encrypted_file == NULL) {
        printf("Failed to open encrypted file: %s\n", filename);
        return;
    }

    char* decrypted_filename = malloc(strlen(filename) - 4);
    strncpy(decrypted_filename, filename, strlen(filename) - 4);

    FILE* decrypted_file = fopen(decrypted_filename, "w");
    if (decrypted_file == NULL) {
        printf("Failed to create decrypted file: %s\n", decrypted_filename);
        fclose(encrypted_file);
        free(decrypted_filename);
        return;
    }

    int key_len = strlen(key);

    char c = fgetc(encrypted_file);
    int i = 0;
    while (c != EOF) {
        fputc(c ^ key[i % key_len], decrypted_file);
        c = fgetc(encrypted_file);
        i++;
    }

    fclose(encrypted_file);
    fclose(decrypted_file);
    free(decrypted_filename);
}

int main() {
    printf("Enter the filename to encrypt: ");
    char filename[256];
    scanf("%s", filename);

    printf("Enter the encryption key: ");
    char key[256];
    scanf("%s", key);

    encrypt(filename, key);
    printf("File encrypted successfully.\n");

    printf("Enter the filename to decrypt: ");
    char encrypted_filename[256];
    scanf("%s", encrypted_filename);

    printf("Enter the decryption key: ");
    char decryption_key[256];
    scanf("%s", decryption_key);

    decrypt(encrypted_filename, decryption_key);
    printf("File decrypted successfully.\n");
    
    return 0;
}
