#ifndef IRRIGATION_SCHEDULE_H
#  define IRRIGATION_SCHEDULE_H

#  include <stdbool.h>
#  include <time.h>
#  include <string.h>
#  include <stdlib.h>
#  include <stdio.h>
#  include <math.h>
#  include <stdarg.h>

#  define MAX_STRING_LENGTH 256
#  define MAX_ZONES 16
#  define MAX_EVENTS 100
#  define SECONDS_IN_MINUTE 60

// Basic device and session structs - defined first as they're used throughout
typedef struct
{
  char duid[MAX_STRING_LENGTH];
  // duid = nondet_char();
  char client_id[MAX_STRING_LENGTH];
  char location_id[MAX_STRING_LENGTH];
  char device_type[MAX_STRING_LENGTH];
  char federated_identity[MAX_STRING_LENGTH];
  time_t user_presence_exp;
} DeviceRecord;

typedef struct
{
  char event[MAX_STRING_LENGTH];
  char zoneId[MAX_STRING_LENGTH];
  char ts[MAX_STRING_LENGTH];
  int duration;
  int seqNum;
  int durMod;
} RunPlanStep;

typedef struct
{
  char event[MAX_STRING_LENGTH];
  char scheduleId[MAX_STRING_LENGTH];
  char zoneId[MAX_STRING_LENGTH];
  char ts[MAX_STRING_LENGTH];
  int durMod;
  int seqNum;
} RunPlanEvent;

typedef struct
{
  char id[MAX_STRING_LENGTH];
  char name[MAX_STRING_LENGTH];
  char type[MAX_STRING_LENGTH];
  char firmware_id[MAX_STRING_LENGTH];
  char client_id[MAX_STRING_LENGTH];
  bool water_sense;
  bool ignore_skip_reason;
  struct
  {
    char zone_uid[MAX_STRING_LENGTH];
    int duration;
    int client_id;
  } zones[MAX_ZONES];
  int num_zones;
  bool cycle_soak;
  int estimated_runtime;
  struct
  {
    char start_at[9];
    char end_before[9];
  } preferred_time;
} IrrigationScheduleRecord;

void handleRunplanEvent(
  DeviceRecord *device,
  RunPlanEvent *events,
  int num_events)
{
  if (!device)
  {
    printf("No usable criterion.\n");
    return;
  }

  if (!events || num_events == 0)
  {
    return;
  }

  verbose("runplan events", events);

  RunPlanEvent *runEvents = (RunPlanEvent *)events;
  IrrigationScheduleRecord *schedule = NULL;
  char error_buf[MAX_STRING_LENGTH];

  if (
    getByFwIdOrThrow(
      runEvents[0].scheduleId, device->client_id, &schedule, error_buf) != 0)
  {
    printf("Error getting schedule: %s\n", error_buf);
    return;
  }

  RunPlanStep *runplanListStep = calloc(num_events, sizeof(RunPlanStep));
  if (!runplanListStep)
  {
    free(schedule);
    return;
  }

  int stepCount = 0;
  for (int i = 0; i < num_events; i++)
  {
    RunPlanEvent *e = &runEvents[i];

    copy_string_safe(
      runplanListStep[stepCount].event, e->event, MAX_STRING_LENGTH);
    copy_string_safe(
      runplanListStep[stepCount].zoneId, e->zoneId, MAX_STRING_LENGTH);
    copy_string_safe(runplanListStep[stepCount].ts, e->ts, MAX_STRING_LENGTH);
    runplanListStep[stepCount].duration = e->durMod;
    runplanListStep[stepCount].seqNum = e->seqNum;
    runplanListStep[stepCount].durMod = e->durMod;

    stepCount++;
  }

  saveExecutionRecord(
    schedule->id, "runplan", runplanListStep, sizeof(RunPlanStep) * stepCount);

  free(runplanListStep);
  free(schedule);
}
