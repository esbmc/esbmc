//GPT-4o-mini DATASET v1.0 Category: File Encryptor ; Style: light-weight
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 1024

void encryptDecrypt(const char *inputFile, const char *outputFile, char key) {
    FILE *input = fopen(inputFile, "rb");
    if (!input) {
        perror("Error opening input file");
        exit(EXIT_FAILURE);
    }

    FILE *output = fopen(outputFile, "wb");
    if (!output) {
        perror("Error opening output file");
        fclose(input);
        exit(EXIT_FAILURE);
    }

    unsigned char buffer[BUFFER_SIZE];
    size_t bytesRead;

    while ((bytesRead = fread(buffer, sizeof(unsigned char), BUFFER_SIZE, input)) > 0) {
        for (size_t i = 0; i < bytesRead; i++) {
            buffer[i] ^= key; // XOR operation for encryption/decryption
        }
        fwrite(buffer, sizeof(unsigned char), bytesRead, output);
    }

    fclose(input);
    fclose(output);
}

void printUsage() {
    printf("Usage: file_encryption <encrypt/decrypt> <input_file> <output_file> <key>\n");
    printf("Example: file_encryption encrypt myfile.txt encrypted.txt 'A'\n");
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printUsage();
        return EXIT_FAILURE;
    }

    const char *operation = argv[1];
    const char *inputFile = argv[2];
    const char *outputFile = argv[3];
    char key = argv[4][0]; // use the first character as the key

    if (strcmp(operation, "encrypt") == 0) {
        encryptDecrypt(inputFile, outputFile, key);
        printf("File encrypted successfully!\n");
    } else if (strcmp(operation, "decrypt") == 0) {
        encryptDecrypt(inputFile, outputFile, key);
        printf("File decrypted successfully!\n");
    } else {
        printUsage();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
