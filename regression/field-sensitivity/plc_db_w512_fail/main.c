/* #7521 field-sensitivity corpus: a PLC data block passed by pointer to the cycle function.
 * The struct has 512 fields; F<n>(p) declares n padding fields. */
#include <assert.h>
#include <stdbool.h>

_Bool nondet_bool(void);

#define F1(p) short p;
#define F2(p) F1(p##0) F1(p##1)
#define F4(p) F2(p##0) F2(p##1)
#define F8(p) F4(p##0) F4(p##1)
#define F16(p) F8(p##0) F8(p##1)
#define F32(p) F16(p##0) F16(p##1)
#define F64(p) F32(p##0) F32(p##1)
#define F128(p) F64(p##0) F64(p##1)
#define F256(p) F128(p##0) F128(p##1)
typedef struct
{
  _Bool start, stop, running, alarm;
  short level, setpoint, count;
  int timer;
  F256(p) F128(q) F64(r) F32(s) F16(t) F8(u)
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
