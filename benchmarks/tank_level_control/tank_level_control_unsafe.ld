<?xml version="1.0" encoding="UTF-8"?>
<!-- CS12-UNSAFE: Industrial Tank Level Control - UNSAFE version -->
<!-- Based on: MathWorks Simulink PLC Coder example (public documentation) -->
<!-- Bug 1: No interlock between PUMP and VALVE — both can run simultaneously -->
<!-- Bug 2: PUMP can run when HIGH_SWITCH=1 (overfill hazard) -->
<!-- Bug 3: VALVE can open when LOW_SWITCH=1 (empty tank hazard) -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="MathWorks-spec" productName="TankLevelControl_Unsafe"
              productVersion="1.0" creationDateTime="2026-06-12"/>
  <contentHeader name="Tank_Level_Control_Unsafe"/>
  <types/>
  <instances>
    <configurations>
      <configuration name="Config">
        <resource name="Resource">
          <task name="MainTask" priority="0" interval="T#10ms"/>
          <pouInstance name="TankCtrl" typeName="Tank_Level_Control_Unsafe"/>
        </resource>
      </configuration>
    </configurations>
  </instances>
  <pous>
    <pou name="Tank_Level_Control_Unsafe" pouType="program">
      <interface>
        <inputVars>
          <variable name="AUTO_MODE"><type><BOOL/></type></variable>
          <variable name="HIGH_SWITCH"><type><BOOL/></type></variable>
          <variable name="LOW_SWITCH"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="PUMP"><type><BOOL/></type></variable>
          <variable name="VALVE"><type><BOOL/></type></variable>
        </outputVars>
        <localVars>
          <variable name="Filling_Active"><type><BOOL/></type></variable>
          <variable name="Draining_Active"><type><BOOL/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <!-- Rung 1: PUMP ON when AUTO_MODE=1 AND LOW_SWITCH=1 -->
          <!-- BUG: no check for HIGH_SWITCH — pump can run when tank is full -->
          <!-- BUG: no interlock with VALVE — both can run simultaneously -->
          <rung localId="1" lineNumber="1">
            <contact variable="AUTO_MODE"/>
            <contact variable="LOW_SWITCH"/>
            <coil variable="PUMP"/>
          </rung>
          <!-- Rung 2: Filling_Active flag -->
          <rung localId="2" lineNumber="2">
            <contact variable="PUMP"/>
            <coil variable="Filling_Active"/>
          </rung>
          <!-- Rung 3: VALVE ON when AUTO_MODE=1 AND HIGH_SWITCH=1 -->
          <!-- BUG: no check for LOW_SWITCH — valve can open when tank is empty -->
          <!-- BUG: no interlock with PUMP — both can run simultaneously -->
          <rung localId="3" lineNumber="3">
            <contact variable="AUTO_MODE"/>
            <contact variable="HIGH_SWITCH"/>
            <coil variable="VALVE"/>
          </rung>
          <!-- Rung 4: Draining_Active flag -->
          <rung localId="4" lineNumber="4">
            <contact variable="VALVE"/>
            <coil variable="Draining_Active"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
