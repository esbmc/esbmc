//FormAI DATASET v0.1 Category: System event logger ; Style: systematic
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Define the maximum length of the log message
#define MAX_LOG_MSG_LEN 100

// Define the structure for the log event
typedef struct LogEvent {
    time_t timestamp;
    char message[MAX_LOG_MSG_LEN];
} LogEvent;

// Define the function to log the event
void logEvent(char* message) {
    // Create the log event and set the timestamp
    LogEvent event;
    event.timestamp = time(NULL);

    // Copy the message into the log event, making sure it doesn't exceed MAX_LOG_MSG_LEN
    int len = snprintf(event.message, MAX_LOG_MSG_LEN, "%s", message);
    if (len >= MAX_LOG_MSG_LEN) {
        event.message[MAX_LOG_MSG_LEN - 1] = '\0';
    }

    // Open the log file for appending
    FILE* logFile = fopen("system.log", "a");
    if (logFile == NULL) {
        printf("Error opening log file.\n");
        return;
    }

    // Write the log event to the file
    fprintf(logFile, "[%ld]: %s\n", event.timestamp, event.message);

    // Close the log file
    fclose(logFile);
}

// Define the main function
int main() {
    // Log some example events
    logEvent("System started.");
    logEvent("User logged in.");
    logEvent("File saved successfully.");
    logEvent("Error: Unable to connect to database.");

    return 0;
}
