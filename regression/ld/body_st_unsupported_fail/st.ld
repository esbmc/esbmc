<?xml version="1.0" encoding="UTF-8"?>
<!-- ST body driving A and B unconditionally: P1 is violated every scan (#7354). -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-08-27"/>
  <contentHeader name="Mutex"/>
  <types/>
  <instances/>
  <pous>
    <pou name="Mutex" pouType="program">
      <interface>
        <outputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <ST><xhtml xmlns="http://www.w3.org/1999/xhtml">A := TRUE;
B := TRUE;
</xhtml></ST>
      </body>
    </pou>
  </pous>
</project>
