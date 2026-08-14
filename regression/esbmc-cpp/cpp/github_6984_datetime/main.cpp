// github.com/esbmc/esbmc/issues/6984 — the aws-sdk-cpp Aws::Utils::DateTime
// shape: a system_clock::time_point representation plus tm members, with
// <chrono> as the only include.
#include <chrono>
#include <cassert>

class DateTime
{
  std::chrono::system_clock::time_point m_time;

public:
  DateTime() = default;
  explicit DateTime(std::chrono::system_clock::time_point t) : m_time(t) {}
  explicit DateTime(int64_t millis)
    : m_time(
        std::chrono::system_clock::time_point(std::chrono::milliseconds(millis)))
  {
  }

  int64_t Millis() const
  {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
             m_time.time_since_epoch())
      .count();
  }

  double SecondsWithMSPrecision() const
  {
    return static_cast<double>(Millis()) / 1000.0;
  }

  DateTime operator+(const std::chrono::milliseconds &d) const
  {
    return DateTime(m_time + d);
  }

  bool operator<(const DateTime &other) const
  {
    return m_time < other.m_time;
  }

  tm GetGmtStruct() const
  {
    tm t = {};
    t.tm_year = 70;
    return t;
  }
};

int main()
{
  DateTime epoch;
  assert(epoch.Millis() == 0);

  DateTime d(1500);
  assert(d.Millis() == 1500);
  assert(d.SecondsWithMSPrecision() > 1.4);

  DateTime later = d + std::chrono::milliseconds(500);
  assert(later.Millis() == 2000);
  assert(d < later);
  assert(!(later < d));

  assert(d.GetGmtStruct().tm_year == 70);
  return 0;
}
