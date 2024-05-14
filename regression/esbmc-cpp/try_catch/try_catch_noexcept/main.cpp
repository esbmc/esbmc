void myfunction () noexcept {
  // throw 5.0;
}

int main (void) {
  try {
    myfunction();
  }
  catch (...) { return 3; }

  return 0;
}