void myfunction () throw (char) {
  throw 'x';
}

int main (void) {
  try {
    myfunction();
  }
  catch (int) { return 3; }
  catch (...) { return 1; }
  catch (char) { return 2; }
  return 0;
}

