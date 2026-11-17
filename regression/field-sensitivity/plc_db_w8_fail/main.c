/* #7521 field-sensitivity corpus: a PLC data block passed by pointer to the cycle function.
 * The struct has 8 fields; F<n>(p) declares n padding fields. */
#include <assert.h>
#include <stdbool.h>

_Bool nondet_bool(void);

typedef struct
{
  _Bool start, stop, running, alarm;
  short level, setpoint, count;
  int timer;
} DB_t;

DB_t db;

void cycle(DB_t *d)
{
  if (d->start && !d->stop)
    d->running = true;
  if (d->stop)
    d->running = false;
  if (d->running)
  {
    d->timer = d->timer + 1;
    if (d->level < d->setpoint)
      d->level = d->level + 1;
    d->count = d->count + 1;
  }
  else
    d->timer = 0;
  d->alarm = d->level > 100;
}

int main(void)
{
  db.setpoint = 50;
  for (int k = 0; k < 20; k++)
  {
    db.start = nondet_bool();
    db.stop = nondet_bool();
    cycle(&db);
    assert(db.timer < 15);
  }
  return 0;
}
