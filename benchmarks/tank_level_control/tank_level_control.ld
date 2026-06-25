<?xml version="1.0" encoding="UTF-8"?>
<!-- CS12: Industrial Tank Level Control -->
<!-- Based on: MathWorks Simulink PLC Coder example -->
<!-- "Model and Generate Ladder Logic Code for Industrial Tank Level Control" -->
<!-- URL: mathworks.com/help/plccoder/ug/model-simulate-and-generate-ladder-logic -->
<!--      -for-industrial-tank-level-control-system.html -->
<!-- Logic specification: PUBLIC (MathWorks documentation) -->
<!-- Variables: AUTO_MODE, HIGH_SWITCH, LOW_SWITCH, PUMP, VALVE -->
<!-- Behaviour: pump fills tank (LOW→HIGH), valve drains (HIGH→LOW) -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="MathWorks-spec" productName="TankLevelControl"
              productVersion="1.0" creationDateTime="2026-06-12"/>
  <contentHeader name="Tank_Level_Control"/>
  <types/>
  <instances>
    <configurations>
      <configuration name="Config">
        <resource name="Resource">
          <task name="MainTask" priority="0" interval="T#10ms"/>
          <pouInstance name="TankCtrl" typeName="Tank_Level_Control"/>
        </resource>
      </configuration>
    </configurations>
  </instances>
  <pous>
    <pou name="Tank_Level_Control" pouType="program">
      <interface>
        <inputVars>
          <!-- AUTO_MODE: 1=automatic control, 0=manual/hold -->
          <variable name="AUTO_MODE"><type><BOOL/></type></variable>
          <!-- HIGH_SWITCH: 1=tank level at or above high setpoint (8) -->
          <variable name="HIGH_SWITCH"><type><BOOL/></type></variable>
          <!-- LOW_SWITCH: 1=tank level at or below low setpoint (2) -->
          <variable name="LOW_SWITCH"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <!-- PUMP: 1=pump running, fills tank -->
          <variable name="PUMP"><type><BOOL/></type></variable>
          <!-- VALVE: 1=drain valve open, empties tank -->
          <variable name="VALVE"><type><BOOL/></type></variable>
        </outputVars>
        <localVars>
          <variable name="Filling_Active"><type><BOOL/></type></variable>
          <variable name="Draining_Active"><type><BOOL/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <!-- Rung 1: PUMP ON when AUTO_MODE=1 AND LOW_SWITCH=1 AND not draining -->
          <!-- "When the level reaches two, the pump turns on and fills the tank" -->
          <rung localId="1" lineNumber="1">
            <contact variable="AUTO_MODE"/>
            <contact variable="LOW_SWITCH"/>
            <contact variable="HIGH_SWITCH" negated="negated"/>
            <contact variable="Draining_Active" negated="negated"/>
            <coil variable="PUMP"/>
          </rung>
          <!-- Rung 2: Filling_Active flag follows PUMP -->
          <rung localId="2" lineNumber="2">
            <contact variable="PUMP"/>
            <coil variable="Filling_Active"/>
          </rung>
          <!-- Rung 3: VALVE ON when AUTO_MODE=1 AND HIGH_SWITCH=1 AND not filling -->
          <!-- "When level exceeds or reaches eight, valve turns on and drains" -->
          <rung localId="3" lineNumber="3">
            <contact variable="AUTO_MODE"/>
            <contact variable="HIGH_SWITCH"/>
            <contact variable="LOW_SWITCH" negated="negated"/>
            <contact variable="Filling_Active" negated="negated"/>
            <coil variable="VALVE"/>
          </rung>
          <!-- Rung 4: Draining_Active flag follows VALVE -->
          <rung localId="4" lineNumber="4">
            <contact variable="VALVE"/>
            <coil variable="Draining_Active"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
