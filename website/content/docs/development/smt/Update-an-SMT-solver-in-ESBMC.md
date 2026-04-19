---
title: SMT Solver Update
---


These instructions provide information on how to update the version of an existing SMT solver in ESBMC.

1) Open the file `BUILDING.md` and update the SMT solver version. For example, we'll update the bitwuzla version from 0.5.0 to 0.6.0.

![image](https://github.com/user-attachments/assets/2c862974-b6a6-4a31-bdb0-d15162aac7a1)

![image](https://github.com/user-attachments/assets/3b1718d5-54d7-4ebb-bbd7-4273b1974405)

2) Open the file `src/solvers/bitwuzla/CMakeLists.txt` and update the SMT solver version (bitwuzla 0.5.0):

![image](https://github.com/user-attachments/assets/cc4849bb-8788-49f6-ad90-326328d89f0d)

3) The next step is to check whether the SMT solver has changed their API signature. If they have changed, we must also update the respective interface in the `src/solvers`


