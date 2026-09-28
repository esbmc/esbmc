#ifndef CPROVER_TIME_STOPPING_H
#define CPROVER_TIME_STOPPING_H

typedef unsigned long long fine_timet;

fine_timet current_time();
void output_time(const fine_timet &fine_time, std::ostream &out);
std::string time2string(const fine_timet &fine_time);

#endif
