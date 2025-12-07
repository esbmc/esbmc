// Test --unwindsetname with mixed scopes: global, namespace, class, static
// Only select functions get explicit bounds

#include <cassert>

// Global function
void global_func() {
  int i, sum = 0;
  for (i = 0; i < 5; i++) {
    sum += i;
  }
  assert(sum == 10);  // 0+1+2+3+4 = 10
}

// Namespace
namespace engine {
  void process() {
    int j, val = 1;
    for (j = 0; j < 4; j++) {
      val *= 2;
    }
    assert(val == 16);  // 2^4 = 16
  }

  void compute() {
    int k, count = 0;
    for (k = 0; k < 3; k++) {
      count++;
    }
    assert(count == 3);
  }
}

// Class
class Worker {
public:
  void task_one() {
    int m, prod = 1;
    for (m = 0; m < 3; m++) {
      prod *= 3;
    }
    assert(prod == 27);  // 3^3 = 27
  }

  void task_two() {
    int n, total = 0;
    for (n = 0; n < 6; n++) {
      total += n;
    }
    assert(total == 15);  // 0+1+2+3+4+5 = 15
  }

  void task_three() {
    int p, result = 0;
    for (p = 0; p < 4; p++) {
      result += p * 2;
    }
    assert(result == 12);  // 0+2+4+6 = 12
  }
};

// Nested: Class inside namespace
namespace core {
  class Processor {
  public:
    void run() {
      int r, val = 0;
      for (r = 0; r < 4; r++) {
        val += r;
      }
      assert(val == 6);  // 0+1+2+3 = 6
    }
  };
}

// Static function (file scope)
static void static_helper() {
  int q, acc = 0;
  for (q = 0; q < 7; q++) {
    acc++;
  }
  assert(acc == 7);
}

int main() {
  // Global - gets explicit bound
  global_func();  // global_func:0:6

  // Namespace - only one gets bound
  engine::process();  // Uses default --unwind 5
  engine::compute();  // N@engine@compute:0:4

  // Class - two get bounds, one uses default
  Worker w;
  w.task_one();    // S@Worker@task_one:0:4
  w.task_two();    // Uses default --unwind 5
  w.task_three();  // S@Worker@task_three:0:5

  // Nested - gets explicit bound with full file@N@namespace@S@class@method syntax
  core::Processor p;
  p.run();  // main.cpp@N@core@S@Processor@run:0:5

  // Static - gets explicit bound with file@ prefix
  static_helper();  // main.cpp@static_helper:0:8

  return 0;
}
