<?xml version="1.0" encoding="UTF-8"?>
<!-- Minimal unsafe program: two coils energised by the same input, no interlock. -->
<!-- Violates mutual exclusion of A and B whenever Enable is TRUE. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-10"/>
  <contentHeader name="MutexFail"/>
  <types/>
  <instances/>
  <pous>
    <pou name="MutexFail" pouType="program">
      <interface>
        <inputVars>
          <variable name="Enable"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <rung localId="1" lineNumber="1">
            <contact variable="Enable"/>
            <coil variable="A"/>
          </rung>
          <rung localId="2" lineNumber="2">
            <contact variable="Enable"/>
            <coil variable="B"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
