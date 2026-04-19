#ifndef IRRIGATION_SCHEDULE_H
#define IRRIGATION_SCHEDULE_H

#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>

#define MAX_STRING_LENGTH 256
#define MAX_ZONES 16
#define MAX_EVENTS 100
#define SECONDS_IN_MINUTE 60

// Basic device and session structs - defined first as they're used throughout
typedef struct {
    char duid[MAX_STRING_LENGTH];
    // duid = nondet_char();
    char client_id[MAX_STRING_LENGTH];
    char location_id[MAX_STRING_LENGTH];
    char device_type[MAX_STRING_LENGTH];
    char federated_identity[MAX_STRING_LENGTH];
    time_t user_presence_exp;
} DeviceRecord;

typedef struct {
    char event[MAX_STRING_LENGTH];
    char zoneId[MAX_STRING_LENGTH];
    char ts[MAX_STRING_LENGTH];
    int duration;
    int seqNum;
    int durMod;
} RunPlanStep;

typedef struct {
    char event[MAX_STRING_LENGTH];
    char scheduleId[MAX_STRING_LENGTH];
    char zoneId[MAX_STRING_LENGTH];
    char ts[MAX_STRING_LENGTH];
    int durMod;
    int seqNum;
} RunPlanEvent;

typedef struct {
    char event[MAX_STRING_LENGTH];
    char status[MAX_STRING_LENGTH];
    char zoneId[MAX_STRING_LENGTH];
    char scheduleId[MAX_STRING_LENGTH];
    int actualDuration;
    int durAct;
    int durDef;
    int plannedDuration;
    char ts[MAX_STRING_LENGTH];  // Changed from time_t to string
} SessionEvent;

typedef struct {
    char event[MAX_STRING_LENGTH];
    char status[MAX_STRING_LENGTH];
    char zoneId[MAX_STRING_LENGTH];
    char ts[MAX_STRING_LENGTH];
    char skipReason[MAX_STRING_LENGTH];
} SessionEventRecord;

// Execution record types
typedef enum {
    EXEC_SESSION = 1,
    EXEC_RUNPLAN = 2,
    EXEC_INIT = 3,
    EXEC_STATE = 4
} ExecutionType;

// Weather related structs
typedef struct {
    float speed;
    float direction;
} Wind;

typedef struct {
    char type[MAX_STRING_LENGTH];
} WeatherType;

typedef struct {
    float temp;
    Wind wind;
    WeatherType weather;
} WeatherCurrent;

typedef struct {
    float pop;
    time_t forecast_time;
} WeatherForecast;

typedef struct {
    int total;
} Precipitation;

typedef struct {
    time_t endTs;
    Precipitation precip;
} WeatherObservation;

typedef struct {
    WeatherCurrent* current;
    WeatherForecast forecast;
    WeatherObservation* observations;
    int observation_count;
    int timezone_offset;
} WeatherData;

// Schedule and session related structs
typedef struct {
    char type[MAX_STRING_LENGTH];
    char id[MAX_STRING_LENGTH];
    struct {
        char status[MAX_STRING_LENGTH];
        int activeSeqNum;
        int durationRemaining;
    } state;
} ExecutionState;

// Forward declaration for ExecutionRecord
typedef struct ExecutionRecord ExecutionRecord;

typedef struct {
    time_t ts;
    char event[MAX_STRING_LENGTH];
    struct {
        char id[MAX_STRING_LENGTH];
        struct {
            char status[MAX_STRING_LENGTH];
            SessionEvent* completed;
            int completed_count;
            RunPlanStep* planned;
            int planned_count;
        } state;
    } body;
} LedgerState;

struct ExecutionRecord {
    char id[MAX_STRING_LENGTH];
    time_t ts;
    ExecutionType type;
    union {
        struct {
            SessionEvent* session;
            int session_count;
        };
        struct {
            RunPlanStep* runplan;
            int runplan_count;
        };
        struct {
            char status[MAX_STRING_LENGTH];
            int activeSeqNum;
            int durationRemaining;
        } state;
    };
};

typedef struct {
    char id[MAX_STRING_LENGTH];
    char name[MAX_STRING_LENGTH];
    char type[MAX_STRING_LENGTH];
    char firmware_id[MAX_STRING_LENGTH];
    char client_id[MAX_STRING_LENGTH];
    bool water_sense;
    bool ignore_skip_reason;
    struct {
        char zone_uid[MAX_STRING_LENGTH];
        int duration;
        int client_id;
    } zones[MAX_ZONES];
    int num_zones;
    bool cycle_soak;
    int estimated_runtime;
    struct {
        char start_at[9];
        char end_before[9];
    } preferred_time;
} IrrigationScheduleRecord;

// Weather control structs
typedef struct {
    float temp_min_threshold;
    float temp_max_threshold;
    float wind_max_threshold;
    float precip_forecast_max_threshold;
} WeatherControls;

typedef struct {
    int day_offset;
    float max;
} PrecipThreshold;

typedef struct {
    WeatherControls controls;
    PrecipThreshold* observed_precipitation_thresholds;
    int threshold_count;
} PreferencesRecord;

typedef struct {
    bool passes;
    char reason[MAX_STRING_LENGTH];
} WeatherControlResult;

typedef struct {
    bool all_skipped;
    int* zone_runtimes;
    int num_zone_runtimes;
} WaterSenseResult;

typedef struct {
    char reason[MAX_STRING_LENGTH];
    char clientId[MAX_STRING_LENGTH];
    time_t until;
} SkipReason;

// Shadow state related structs
typedef struct {
    char status[MAX_STRING_LENGTH];
    int runPlanSeqNum;
    int durationRemaining;
    bool paused;
} HydraOverview;

typedef struct {
    char runningID[MAX_STRING_LENGTH];
    char nextID[MAX_STRING_LENGTH];
    bool nextWillRunSoon;
    char nextStartTime[MAX_STRING_LENGTH];
} ScheduleInfo;

typedef struct {
    HydraOverview* hydraOverview;
    ScheduleInfo* schedule;
} ShadowReportedState;

typedef struct {
    ShadowReportedState* reported;
} ShadowStateObject;

typedef struct {
    ShadowStateObject* current;
    ShadowStateObject* previous;
    ShadowStateObject* delta;
} shadowState;

// Function declarations
void handleRunplanEvent(DeviceRecord* device, RunPlanEvent* events, int num_events);
void handleSessionEvent(DeviceRecord* device, void* events, int num_events);
void handleShadowUpdate(DeviceRecord* device, void* current, void* previous);
void estimateScheduleRun(const char* duid, IrrigationScheduleRecord* schedule);
bool tryUpdateExecution(DeviceRecord* device, shadowState* ss);
IrrigationScheduleRecord* getNextSchedule(DeviceRecord* device, shadowState* ss);
void weatherSkipCheck(DeviceRecord* device, shadowState* ss, IrrigationScheduleRecord* schedule);
void waterSenseCheck(DeviceRecord* device, shadowState* ss, IrrigationScheduleRecord* schedule);
WeatherControlResult passesWeatherControls(DeviceRecord* device);
void notifyScheduleUpdate(DeviceRecord* device, IrrigationScheduleRecord* schedule);

// Helper functions
time_t ttlDate(int offset_seconds);
void* rollupLedger(IrrigationScheduleRecord* schedule);
void verbose(const char* message, ...);
const char* ovStatus(void* overview);
int getByFwIdOrThrow(const char* fwid, const char* clientId, 
                     IrrigationScheduleRecord** schedule, char* error_buf);
bool hasChanged(shadowState* ss, const char* (*getter)(const ShadowReportedState*));
bool mqttPublish(const char* topic, const char* payload);
time_t dateToTs(const char* date_str);
ExecutionRecord* getExecutionRecords(const char* scheduleId);
const char* getScheduleRunningID(const ShadowReportedState* state);
const char* getScheduleNextWillRunSoon(const ShadowReportedState* state);
const char* getScheduleNextStartTime(const ShadowReportedState* state);
void saveExecutionRecord(const char* id, const char* type, const void* data, size_t data_size);
void updateDeviceDesired(const char* clientId, const SkipReason* skip);

// String handling functions
static inline void copy_string_safe(char* dest, const char* src, size_t size) {
    if (!dest || !src || size == 0) return;
    strncpy(dest, src, size - 1);
    dest[size - 1] = '\0';
}

static inline bool is_string_empty(const char* str) {
    return !str || str[0] == '\0';
}

#endif // IRRIGATION_SCHEDULE_H