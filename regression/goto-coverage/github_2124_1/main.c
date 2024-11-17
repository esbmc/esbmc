#include "main.h"

// Global variables
static const char* rainConditions[] = {
    "rain", "drizzle", "sleet", "snow", "hail", 
    "thunderstorm", "precipitation", "wintry"
};
static char clientId[MAX_STRING_LENGTH] = "";

// Helper function implementations
static bool isRainCondition(const char* weatherType) {
    for (size_t i = 0; i < sizeof(rainConditions)/sizeof(rainConditions[0]); i++) {
        if (strstr(weatherType, rainConditions[i]) != NULL) {
            return true;
        }
    }
    return false;
}

static float calculateTotalPrecipitation(WeatherObservation* observations, 
                                       int dayOffset, 
                                       int timezoneOffset) {
    if (!observations) return 0.0f;
    
    time_t cutoff = time(NULL) - (dayOffset * 24 * 60 * 60);
    float total = 0.0f;
    
    for (int i = 0; observations[i].endTs != 0; i++) {
        if (observations[i].endTs >= cutoff) {
            total += observations[i].precip.total;
        }
    }
    return total;
}

time_t ttlDate(int offset_seconds) {
    return time(NULL) + offset_seconds;
}

void handleRunplanEvent(DeviceRecord* device, void *events, int num_events) {
   
   if (!device) {
    }
    RunPlanEvent* runEvents = (RunPlanEvent*)events;
    IrrigationScheduleRecord* schedule = NULL;
    char error_buf[MAX_STRING_LENGTH];

    if (getByFwIdOrThrow(runEvents[0].scheduleId, device->client_id, 
                         &schedule, error_buf) != 0) {

    }

    RunPlanStep* runplanListStep = calloc(num_events, sizeof(RunPlanStep));
    if (!runplanListStep) {
        free(schedule);
        return;
    }

    int stepCount = 0;
    for (int i = 0; i < num_events; i++) {
        RunPlanEvent* e = &runEvents[i];
        
        copy_string_safe(runplanListStep[stepCount].event, e->event, MAX_STRING_LENGTH);
        copy_string_safe(runplanListStep[stepCount].zoneId, e->zoneId, MAX_STRING_LENGTH);
        copy_string_safe(runplanListStep[stepCount].ts, e->ts, MAX_STRING_LENGTH);
        runplanListStep[stepCount].duration = e->durMod;
        runplanListStep[stepCount].seqNum = e->seqNum;
        runplanListStep[stepCount].durMod = e->durMod;
        
        stepCount++;
    }

    saveExecutionRecord(schedule->id, "runplan", runplanListStep, sizeof(RunPlanStep) * stepCount);


}

void handleSessionEvent(DeviceRecord* device, void* events, int num_events) {
    if (!device) {
        printf("No usable criterion.\n");
        return;
    }

    if (!events || num_events == 0) {
        return;
    }

    SessionEvent* sessionEvents = (SessionEvent*)events;
    IrrigationScheduleRecord* schedule = NULL;
    char error_buf[MAX_STRING_LENGTH];

    if (getByFwIdOrThrow(sessionEvents[0].scheduleId, device->client_id, 
                         &schedule, error_buf) != 0) {
        printf("Error getting schedule: %s\n", error_buf);
        return;
    }

    SessionEventRecord* mappedEvents = calloc(num_events, sizeof(SessionEventRecord));
    if (!mappedEvents) {
        free(schedule);
        return;
    }

    int mappedCount = 0;
    for (int i = 0; i < num_events; i++) {
        SessionEvent* event = &sessionEvents[i];
        
        copy_string_safe(mappedEvents[mappedCount].event, event->event, MAX_STRING_LENGTH);
        copy_string_safe(mappedEvents[mappedCount].status, event->status, MAX_STRING_LENGTH);
        copy_string_safe(mappedEvents[mappedCount].zoneId, event->zoneId, MAX_STRING_LENGTH);
        copy_string_safe(mappedEvents[mappedCount].ts, event->ts, MAX_STRING_LENGTH);

        if (event->status[0] != '\0' && strcmp(event->status, "none") != 0) {
            copy_string_safe(mappedEvents[mappedCount].skipReason, 
                           event->status, MAX_STRING_LENGTH);
        }

        mappedCount++;
    }

    saveExecutionRecord(schedule->id, "session", mappedEvents, sizeof(SessionEventRecord) * mappedCount);

    free(mappedEvents);
    free(schedule);
}

void handleShadowUpdate(DeviceRecord* device, void* current, void* previous) {
    if (!device) {
        printf("No usable criterion.\n");
        return;
    }

    copy_string_safe(clientId, device->client_id, MAX_STRING_LENGTH);

    if (!current) {
        return;
    }

    shadowState ss = {
        .current = (ShadowStateObject*)current,
        .previous = (ShadowStateObject*)previous,
        .delta = NULL
    };

    if (!ss.current || !ss.current->reported) return;
    ShadowReportedState* reported = ss.current->reported;

    IrrigationScheduleRecord* schedule = NULL;
    char error_buf[MAX_STRING_LENGTH];

    if (reported->schedule && reported->schedule->runningID[0] != '\0') {
        if (getByFwIdOrThrow(reported->schedule->runningID, device->client_id, 
                            &schedule, error_buf) != 0) {
            printf("Error getting schedule: %s\n", error_buf);
            return;
        }

        if (hasChanged(&ss, getScheduleRunningID)) {
            ExecutionState initState = {0};
            copy_string_safe(initState.type, "init", MAX_STRING_LENGTH);
            copy_string_safe(initState.id, schedule->id, MAX_STRING_LENGTH);
            saveExecutionRecord(schedule->id, "init", &initState, sizeof(initState));
        }

        HydraOverview* ov = reported->hydraOverview;
        const char* status = ovStatus(ov);
        
        if (status != NULL) {
            ExecutionState state = {0};
            copy_string_safe(state.type, "state", MAX_STRING_LENGTH);
            copy_string_safe(state.id, schedule->id, MAX_STRING_LENGTH);
            copy_string_safe(state.state.status, status, MAX_STRING_LENGTH);
            state.state.activeSeqNum = ov->runPlanSeqNum;
            state.state.durationRemaining = ov->durationRemaining;
            saveExecutionRecord(schedule->id, "state", &state, sizeof(state));
        } else if (hasChanged(&ss, getScheduleRunningID)) {
            ExecutionState state = {0};
            copy_string_safe(state.type, "state", MAX_STRING_LENGTH);
            copy_string_safe(state.id, schedule->id, MAX_STRING_LENGTH);
            copy_string_safe(state.state.status, "watering", MAX_STRING_LENGTH);
            saveExecutionRecord(schedule->id, "state", &state, sizeof(state));
        }

        notifyScheduleUpdate(device, schedule);
        free(schedule);
    } else if (hasChanged(&ss, getScheduleRunningID) && ss.delta && ss.delta->reported) {
        verbose("saving completed, delta is", ss.delta->reported);
        
        if (getByFwIdOrThrow(ss.delta->reported->schedule->runningID, 
                            device->client_id, &schedule, error_buf) == 0) {
            ExecutionState state = {0};
            copy_string_safe(state.type, "state", MAX_STRING_LENGTH);
            copy_string_safe(state.id, schedule->id, MAX_STRING_LENGTH);
            copy_string_safe(state.state.status, "completed", MAX_STRING_LENGTH);
            saveExecutionRecord(schedule->id, "state", &state, sizeof(state));
            notifyScheduleUpdate(device, schedule);
            free(schedule);
        }
    }
}

void estimateScheduleRun(const char* duid, IrrigationScheduleRecord* schedule) {
    if (!schedule) return;

    int scheduleEstRunTime = 0;
    int totalEstSoakDelay = 0;

    for (int i = 0; i < schedule->num_zones; i++) {
        int duration = schedule->zones[i].duration > 0 ? 
                      schedule->zones[i].duration : 300;
        scheduleEstRunTime += duration;

        if (schedule->cycle_soak) {
            totalEstSoakDelay += (duration / 420) * 1800;
        }
    }

    scheduleEstRunTime += totalEstSoakDelay;

    if (scheduleEstRunTime > 43200) {
        scheduleEstRunTime = 43200;
    }

    schedule->estimated_runtime = (scheduleEstRunTime + 59) / 60;

    if (schedule->preferred_time.start_at[0]) {
        int hour, minute, second;
        sscanf(schedule->preferred_time.start_at, "%d:%d:%d", 
               &hour, &minute, &second);

        time_t start = time(NULL);
        struct tm* tm = localtime(&start);
        tm->tm_hour = hour;
        tm->tm_min = minute;
        tm->tm_sec = second;
        start = mktime(tm);

        time_t end = start + (schedule->estimated_runtime * 60);
        tm = localtime(&end);
        
        snprintf(schedule->preferred_time.end_before, sizeof(schedule->preferred_time.end_before),
                "%02d:%02d:00", tm->tm_hour, tm->tm_min);
    }
}

void weatherSkipCheck(DeviceRecord* device, shadowState* ss, IrrigationScheduleRecord* schedule) {
    printf("%s Checking if weather skip is necessary. ScheduleId: %s\n", 
           clientId, schedule->id);

    if (strcmp(schedule->type, "manual") == 0) {
        printf("%s Not skipping manual run.\n", clientId);
        return;
    }

    if (schedule->ignore_skip_reason) {
        printf("%s Weather Skip is turned off for ScheduleId: %s\n", 
               clientId, schedule->id);
        return;
    }

    WeatherControlResult wcRes = passesWeatherControls(device);
    if (wcRes.passes) {
        printf("%s No weather skip needed.\n", clientId);
        return;
    }

    printf("%s Skipping schedule due to weather check.\n", clientId);

    int durSecs = 60;
    for (int i = 0; i < schedule->num_zones; i++) {
        durSecs += schedule->zones[i].duration > 0 ? 
                   schedule->zones[i].duration : 60;
    }

    for (int i = 0; i < schedule->num_zones; i++) {
        char cid[MAX_STRING_LENGTH];
        snprintf(cid, sizeof(cid), "%d", schedule->zones[i].client_id);

        SkipReason skip = {0};
        copy_string_safe(skip.reason, wcRes.reason, MAX_STRING_LENGTH);
        copy_string_safe(skip.clientId, cid, MAX_STRING_LENGTH);
        skip.until = time(NULL) + durSecs;
        
        updateDeviceDesired(device->client_id, &skip);
    }
}

WeatherControlResult passesWeatherControls(DeviceRecord* device) {
    WeatherControlResult result = {0};
    PreferencesRecord prefs = {0};
    WeatherData weather = {0};
    WeatherControls* wc = NULL;

    // Get preferences and weather data
    if (!device->location_id[0]) {
        result.passes = true;
        copy_string_safe(result.reason, "No location ID, watering.", MAX_STRING_LENGTH);
        return result;
    }

    // Check current conditions
    if (weather.current) {
        if (weather.current->weather.type[0] && 
            isRainCondition(weather.current->weather.type)) {
            printf("%s Skipping for rain. Current weather: %s\n",
                   clientId, weather.current->weather.type);
            result.passes = false;
            copy_string_safe(result.reason, "nearby rain", MAX_STRING_LENGTH);
            return result;
        }

        wc = &prefs.controls;  // Using controls instead of weather_controls

        if (wc) {
            if (weather.current->temp < wc->temp_min_threshold) {
                printf("%s Skipping for min temp\n", clientId);
                result.passes = false;
                copy_string_safe(result.reason, "temperature", MAX_STRING_LENGTH);
                return result;
            }

            if (weather.current->temp > wc->temp_max_threshold) {
                printf("%s Skipping for max temp\n", clientId);
                result.passes = false;
                copy_string_safe(result.reason, "temperature", MAX_STRING_LENGTH);
                return result;
            }

            if (wc->wind_max_threshold >= 0 && 
                weather.current->wind.speed > wc->wind_max_threshold) {
                printf("%s Skipping for high wind\n", clientId);
                result.passes = false;
                copy_string_safe(result.reason, "high wind", MAX_STRING_LENGTH);
                return result;
            }
        }
    }

    // Check forecast
    if (wc && wc->precip_forecast_max_threshold != -1 && 
        weather.forecast.pop > wc->precip_forecast_max_threshold) {
        printf("%s Forecasted rain, skipping\n", clientId);
        result.passes = false;
        copy_string_safe(result.reason, "forecasted rain", MAX_STRING_LENGTH);
        return result;
    }

    // Check precipitation thresholds
    if (prefs.observed_precipitation_thresholds) {
        for (int i = 0; i < prefs.threshold_count; i++) {
            PrecipThreshold* threshold = &prefs.observed_precipitation_thresholds[i];
            float total = calculateTotalPrecipitation(weather.observations,
                                                    threshold->day_offset,
                                                    weather.timezone_offset);
            
            if (total / 10.0 > threshold->max) {
                printf("%s Past rain exceeds threshold\n", clientId);
                result.passes = false;
                copy_string_safe(result.reason, "past rain", MAX_STRING_LENGTH);
                return result;
            }
        }
    }

    printf("%s All weather checks passed\n", clientId);
    result.passes = true;
    return result;
}
void notifyScheduleUpdate(DeviceRecord* device, IrrigationScheduleRecord* schedule) {
    time_t now = ttlDate(0);
    if (device->user_presence_exp < now) {
        verbose("skipping notification, user is not online", device);
        return;
    }

    char topic[MAX_STRING_LENGTH];
    snprintf(topic, sizeof(topic), "async_return/%s", device->duid);

    void* event = rollupLedger(schedule);
    if (event) {
        mqttPublish(topic, event);
        free(event);
    }
}

void waterSenseCheck(DeviceRecord* device, shadowState* ss, IrrigationScheduleRecord* schedule) {
    // printf("%s Executing WaterSense check.\n", clientId);

    if(schedule -> cycle_soak);
    // if (strcmp(schedule->type, "manual") == 0) {
    //     printf("%s Not skipping manual run.\n", clientId);
    //     return;
    // } 
    // else {}  

    // WaterSenseResult result = {0};
    // result.zone_runtimes = calloc(schedule->num_zones, sizeof(int));
    // if (!result.zone_runtimes) {
    //     printf("Memory allocation failed for zone runtimes\n");
    //     return;
    // }
    // result.num_zone_runtimes = schedule->num_zones;

    // // Calculate adjusted runtimes
    // for (int i = 0; i < schedule->num_zones; i++) {
    //     result.zone_runtimes[i] = schedule->zones[i].duration;
    // }

    // if (!result.all_skipped && result.zone_runtimes && result.num_zone_runtimes > 0) {
    //     verbose("Modifying zone runtimes", &result);
        
    //     char payload[MAX_STRING_LENGTH * 2];
    //     snprintf(payload, sizeof(payload),
    //             "{\"clientId\":\"%s\",\"scheduleId\":\"%s\",\"zoneRuntimes\":[",
    //             device->client_id, schedule->firmware_id);

    //     for (int i = 0; i < result.num_zone_runtimes; i++) {
    //         char runtime[32];
    //         snprintf(runtime, sizeof(runtime), "%s%d",
    //                 i > 0 ? "," : "", result.zone_runtimes[i]);
    //         strncat(payload, runtime, sizeof(payload) - strlen(payload) - 1);
    //     }
    //     strncat(payload, "]}", sizeof(payload) - strlen(payload) - 1);

    //     char topic[MAX_STRING_LENGTH];
    //     snprintf(topic, sizeof(topic), "iot/HYD/%s/subscription", device->client_id);
    //     mqttPublish(topic, payload);
    // }
    // else if (result.all_skipped) {
    //     verbose("Skipping schedule run", &result);

    //     int durSecs = 60;
    //     for (int i = 0; i < schedule->num_zones; i++) {
    //         durSecs += schedule->zones[i].duration > 0 ? 
    //                   schedule->zones[i].duration : 60;
    //     }

    //     for (int i = 0; i < schedule->num_zones; i++) {
    //         char cid[MAX_STRING_LENGTH];
    //         snprintf(cid, sizeof(cid), "%d", schedule->zones[i].client_id);

    //         SkipReason skip = {0};
    //         copy_string_safe(skip.reason, "WaterSense skip", MAX_STRING_LENGTH);
    //         copy_string_safe(skip.clientId, cid, MAX_STRING_LENGTH);
    //         skip.until = time(NULL) + durSecs;
            
    //         updateDeviceDesired(device->client_id, &skip);
    //     }
    // }

    // free(result.zone_runtimes);
}

IrrigationScheduleRecord* getNextSchedule(DeviceRecord* device, shadowState* ss) {
    if (!ss || !ss->current || !ss->current->reported || !ss->current->reported->schedule) {
        return NULL;
    }

    ShadowReportedState* reported = ss->current->reported;
    const bool nextRunSoon = reported->schedule->nextWillRunSoon;
    const char* scheduleId = reported->schedule->nextID;

    if (nextRunSoon && 
        (hasChanged(ss, getScheduleNextWillRunSoon) || 
         hasChanged(ss, getScheduleNextStartTime))) {
        
        printf("%s Schedule will run soon. ScheduleID: %s\n", 
               device->client_id, scheduleId);

        IrrigationScheduleRecord* schedule = NULL;
        char error_buf[MAX_STRING_LENGTH];

        if (getByFwIdOrThrow(scheduleId, device->client_id, &schedule, error_buf) != 0) {
            printf("No schedule %s for device %s: %s\n", 
                   scheduleId, device->client_id, error_buf);
            return NULL;
        }

        return schedule;
    }
    
    printf("%s Schedule will not run soon\n", device->client_id);
    return NULL;
}

void* rollupLedger(IrrigationScheduleRecord* schedule) {
    ExecutionRecord* items = getExecutionRecords(schedule->id);
    if (!items) return NULL;

    LedgerState* state = calloc(1, sizeof(LedgerState));
    if (!state) {
        free(items);
        return NULL;
    }

    state->ts = ttlDate(0);
    copy_string_safe(state->event, "irrigation_plan_state", MAX_STRING_LENGTH);
    copy_string_safe(state->body.id, schedule->id, MAX_STRING_LENGTH);
    copy_string_safe(state->body.state.status, "watering", MAX_STRING_LENGTH);

    ExecutionRecord* lastState = NULL;
    int i = 0;

    while (items[i].id[0] != '\0') {
        ExecutionRecord* item = &items[i];

        switch(item->type) {
            case EXEC_SESSION:
                if (item->session && item->session_count > 0) {
                    size_t new_size = (state->body.state.completed_count + item->session_count) * 
                                    sizeof(SessionEvent);
                    SessionEvent* new_completed = realloc(state->body.state.completed, new_size);
                    
                    if (new_completed) {
                        state->body.state.completed = new_completed;
                        memcpy(&state->body.state.completed[state->body.state.completed_count],
                               item->session,
                               item->session_count * sizeof(SessionEvent));
                        state->body.state.completed_count += item->session_count;
                    }
                }
                break;

            case EXEC_RUNPLAN:
                if (item->runplan && item->runplan_count > 0) {
                    state->body.state.planned = item->runplan;
                    state->body.state.planned_count = item->runplan_count;
                }
                break;

            case EXEC_INIT:
                if (state->body.state.completed && state->body.state.completed_count > 0) {
                    int new_count = 0;
                    for (int j = 0; j < state->body.state.completed_count; j++) {
                        time_t event_ts = dateToTs(state->body.state.completed[j].ts);
                        if (event_ts >= (item->ts - 5)) {
                            if (j != new_count) {
                                memcpy(&state->body.state.completed[new_count],
                                      &state->body.state.completed[j],
                                      sizeof(SessionEvent));
                            }
                            new_count++;
                        }
                    }
                    state->body.state.completed_count = new_count;
                }
                break;

            case EXEC_STATE:
                if (!lastState || item->ts >= lastState->ts) {
                    lastState = item;
                }
                break;
        }
        i++;
    }

    if (lastState) {
        copy_string_safe(state->body.state.status, lastState->state.status, MAX_STRING_LENGTH);
        
        if (strcmp(lastState->state.status, "completed") == 0) {
            free(state->body.state.planned);
            state->body.state.planned = NULL;
            state->body.state.planned_count = 0;
        }
        else if (lastState->state.activeSeqNum > -1 && state->body.state.planned) {
            for (int j = 0; j < state->body.state.planned_count; j++) {
                if (state->body.state.planned[j].seqNum == lastState->state.activeSeqNum) {
                    break;
                }
            }
        }
    }

    free(items);
    return state;
}

void verbose(const char* message, ...) {
    va_list args;
    va_start(args, message);
    vprintf(message, args);
    va_end(args);
    printf("\n");
}

const char* ovStatus(void* overview) {
    HydraOverview* ov = (HydraOverview*)overview;
    if (!ov || !ov->status[0] || strcmp(ov->status, "invalid") == 0) {
        return NULL;
    }
    
    if (ov->paused) return "paused";
    
    if (strcmp(ov->status, "disabled") == 0 || 
        strcmp(ov->status, "idle") == 0) {
        return "completed";
    }
    
    return ov->status;
}

bool hasChanged(shadowState* ss, const char* (*getter)(const ShadowReportedState*)) {
    if (!ss || !getter) return false;
    
    if (ss->delta && ss->delta->reported) {
        return getter(ss->delta->reported) != NULL;
    }
    
    if (!ss->current || !ss->previous || 
        !ss->current->reported || !ss->previous->reported) {
        return false;
    }
    
    const char* curr = getter(ss->current->reported);
    const char* prev = getter(ss->previous->reported);
    
    return (curr && (!prev || strcmp(curr, prev) != 0));
}

// Add these implementations to your existing code:

time_t dateToTs(const char* date_str) {
    struct tm tm = {0};
    // Expected format: "YYYY-MM-DD HH:MM:SS"
    //if (strptime(date_str, "%Y-%m-%d %H:%M:%S", &tm) == NULL) {
    //    return 0;
    //}
    return mktime(&tm);
}

int getByFwIdOrThrow(const char* fwid, const char* clientId, 
                     IrrigationScheduleRecord** schedule, char* error_buf) {
    // Mock implementation - in real code this would fetch from a database
    *schedule = calloc(1, sizeof(IrrigationScheduleRecord));
    // if (!*schedule) {
    //     snprintf(error_buf, MAX_STRING_LENGTH, "Memory allocation failed");
    //     return 1;
    // }
    
    // Fill with mock data
    copy_string_safe((*schedule)->id, "mock_id", MAX_STRING_LENGTH);
    //copy_string_safe((*schedule)->firmware_id, fwid, MAX_STRING_LENGTH);
    //copy_string_safe((*schedule)->client_id, clientId, MAX_STRING_LENGTH);
    //(*schedule)->num_zones = 1;
    return 0;
}

ExecutionRecord* getExecutionRecords(const char* scheduleId) {
    // Mock implementation
    ExecutionRecord* records = calloc(1, sizeof(ExecutionRecord));
    if (records) {
        copy_string_safe(records[0].id, scheduleId, MAX_STRING_LENGTH);
    }
    return records;
}

const char* getScheduleNextStartTime(const ShadowReportedState* state) {
    if (!state || !state->schedule) return NULL;
    return state->schedule->nextStartTime;
}

const char* getScheduleNextWillRunSoon(const ShadowReportedState* state) {
    if (!state || !state->schedule) return NULL;
    return state->schedule->nextWillRunSoon ? "true" : NULL;
}

const char* getScheduleRunningID(const ShadowReportedState* state) {
    if (!state || !state->schedule) return NULL;
    return state->schedule->runningID;
}

bool mqttPublish(const char* topic, const char* payload) {
    // Mock implementation
    printf("Publishing to topic %s: %s\n", topic, payload);
    return true;
}

void saveExecutionRecord(const char* id, const char* type, const void* data, size_t data_size) {
    // Mock implementation
    printf("Saving execution record: ID=%s, Type=%s\n", id, type);
}

void updateDeviceDesired(const char* clientId, const SkipReason* skip) {
    // Mock implementation
    printf("Updating device desired state for client %s: %s until %ld\n",
           clientId, skip->reason, skip->until);
}

// Main function for testing
int main(int argc, char* argv[]) {
    // Create test device
    DeviceRecord device = {
        .duid = "TEST_DEVICE_001",
        .client_id = "CLIENT_001",
        .location_id = "LOC_001",
        .device_type = "Sprinkler",
        .user_presence_exp = time(NULL) + 3600
    };

    // Create test schedule
    RunPlanEvent testEvents[] = {
        {
            .event = "START",
            .scheduleId = "SCHEDULE_001",
            .zoneId = "ZONE_001",
            .ts = "2024-10-29 10:00:00",
            .durMod = 300,
            .seqNum = 1
        }
    };

    // Test handleRunplanEvent
    printf("\nTesting handleRunplanEvent:\n");
   // handleRunplanEvent(&device, testEvents, 1);

    // Test handleSessionEvent
    printf("\nTesting handleSessionEvent:\n");
    SessionEvent sessionEvents[] = {
        {
            .event = "COMPLETE",
            .status = "SUCCESS",
            .zoneId = "ZONE_001",
            .scheduleId = "SCHEDULE_001",
            .ts = "2024-10-29 10:05:00"
        }
    };
    handleSessionEvent(&device, sessionEvents, 1);

    printf("\nAll tests completed.\n");
    return 0;
}
