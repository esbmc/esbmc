<?xml version="1.0" encoding="UTF-8"?>
<!-- CS1: Three-phase motor forward/reverse interlock (PLCopen XML format) -->
<!-- Safety property: Forward and Reverse coils must never be simultaneously energised -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-09"/>
  <contentHeader name="MotorInterlock"/>
  <types/>
  <instances>
    <configurations>
      <configuration name="Config">
        <resource name="Resource">
          <task name="MainTask" priority="0" interval="T#10ms"/>
          <pouInstance name="MotorInterlock_POU" typeName="MotorInterlock"/>
        </resource>
      </configuration>
    </configurations>
  </instances>
  <pous>
    <pou name="MotorInterlock" pouType="program">
      <interface>
        <inputVars>
          <variable name="Start_Fwd"><type><BOOL/></type></variable>
          <variable name="Start_Rev"><type><BOOL/></type></variable>
          <variable name="Stop"><type><BOOL/></type></variable>
          <variable name="OL_Trip"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="Motor_Forward"><type><BOOL/></type></variable>
          <variable name="Motor_Reverse"><type><BOOL/></type></variable>
        </outputVars>
        <localVars>
          <variable name="Fwd_Seal"><type><BOOL/></type></variable>
          <variable name="Rev_Seal"><type><BOOL/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <!-- Rung 1: Forward direction with hardware interlock (NC contact on Motor_Reverse) -->
          <rung localId="1" lineNumber="1">
            <contact variable="Start_Fwd"/>
            <contact variable="Motor_Reverse" negated="negated"/>
            <contact variable="Stop" negated="negated"/>
            <contact variable="OL_Trip" negated="negated"/>
            <coil variable="Motor_Forward"/>
          </rung>
          <!-- Rung 2: Reverse direction with hardware interlock (NC contact on Motor_Forward) -->
          <rung localId="2" lineNumber="2">
            <contact variable="Start_Rev"/>
            <contact variable="Motor_Forward" negated="negated"/>
            <contact variable="Stop" negated="negated"/>
            <contact variable="OL_Trip" negated="negated"/>
            <coil variable="Motor_Reverse"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
