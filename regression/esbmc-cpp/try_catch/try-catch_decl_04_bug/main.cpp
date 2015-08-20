void myfunction () throw () {
  throw 5.0;
}

int main (void) {
  try {
    myfunction();
  }
  catch (int) { return 3; }
  catch (char) { return 2; }
  catch (...) { return 1; }
  return 0;
}
