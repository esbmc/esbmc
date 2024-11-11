

#include "main.h"

void handleRunplanEvent(DeviceRecord *device, void *events, int num_events)
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

  // verbose("runplan events", events);

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

int main()
{
}