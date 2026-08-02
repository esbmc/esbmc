<?xml version="1.0" encoding="UTF-8"?>
<!-- Negative test: forward/reverse coils driven directly with NO interlock.   -->
<!-- Both inputs can be TRUE in the same scan, so Motor_Forward and            -->
<!-- Motor_Reverse can be energised simultaneously, violating P1.             -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-10"/>
  <contentHeader name="MissingInterlock"/>
  <types/>
  <instances>
    <configurations>
      <configuration name="Config">
        <resource name="Resource">
          <task name="MainTask" priority="0" interval="T#10ms"/>
          <pouInstance name="MissingInterlock_POU" typeName="MissingInterlock"/>
        </resource>
      </configuration>
    </configurations>
  </instances>
  <pous>
    <pou name="MissingInterlock" pouType="program">
      <interface>
        <inputVars>
          <variable name="Start_Fwd"><type><BOOL/></type></variable>
          <variable name="Start_Rev"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="Motor_Forward"><type><BOOL/></type></variable>
          <variable name="Motor_Reverse"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <!-- Rung 1: forward coil, no interlock contact -->
          <rung localId="1" lineNumber="1">
            <contact variable="Start_Fwd"/>
            <coil variable="Motor_Forward"/>
          </rung>
          <!-- Rung 2: reverse coil, no interlock contact -->
          <rung localId="2" lineNumber="2">
            <contact variable="Start_Rev"/>
            <coil variable="Motor_Reverse"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
