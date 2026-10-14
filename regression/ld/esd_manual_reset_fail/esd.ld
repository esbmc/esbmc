<?xml version="1.0" encoding="UTF-8"?>
<!-- CS3: Emergency Shutdown System (ESD) for process plant (SIL-2) -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-09"/>
  <contentHeader name="EmergencyShutdown"/>
  <types/>
  <instances>
    <configurations>
      <configuration name="Config">
        <resource name="Resource">
          <task name="MainTask" priority="0" interval="T#10ms"/>
          <pouInstance name="ESD_POU" typeName="EmergencyShutdown"/>
        </resource>
      </configuration>
    </configurations>
  </instances>
  <pous>
    <pou name="EmergencyShutdown" pouType="program">
      <interface>
        <inputVars>
          <variable name="High_Pressure_Alarm"><type><BOOL/></type></variable>
          <variable name="High_Temp_Alarm"><type><BOOL/></type></variable>
          <variable name="Manual_ESD"><type><BOOL/></type></variable>
          <variable name="Reset_Command"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="ESD_Valve_Closed"><type><BOOL/></type></variable>
          <variable name="Process_Shutdown"><type><BOOL/></type></variable>
          <variable name="Alarm_Active"><type><BOOL/></type></variable>
        </outputVars>
        <localVars>
          <variable name="ESD_Latch"><type><BOOL/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <!-- Rung 1: ESD trigger — any alarm or manual trip sets the latch -->
          <rung localId="1" lineNumber="1">
            <contact variable="High_Pressure_Alarm"/>
            <coil kind="set" variable="ESD_Latch"/>
          </rung>
          <rung localId="2" lineNumber="2">
            <contact variable="High_Temp_Alarm"/>
            <coil kind="set" variable="ESD_Latch"/>
          </rung>
          <rung localId="3" lineNumber="3">
            <contact variable="Manual_ESD"/>
            <coil kind="set" variable="ESD_Latch"/>
          </rung>
          <!-- Rung 4: Reset only when commanded AND no alarms present -->
          <rung localId="4" lineNumber="4">
            <contact variable="Reset_Command"/>
            <contact variable="High_Pressure_Alarm" negated="negated"/>
            <contact variable="High_Temp_Alarm" negated="negated"/>
            <coil kind="reset" variable="ESD_Latch"/>
          </rung>
          <!-- Rung 5: Close valve when latched -->
          <rung localId="5" lineNumber="5">
            <contact variable="ESD_Latch"/>
            <coil variable="ESD_Valve_Closed"/>
          </rung>
          <!-- Rung 6: Shutdown when latched -->
          <rung localId="6" lineNumber="6">
            <contact variable="ESD_Latch"/>
            <coil variable="Process_Shutdown"/>
          </rung>
          <!-- Rung 7: Alarm output -->
          <rung localId="7" lineNumber="7">
            <contact variable="High_Pressure_Alarm"/>
            <coil variable="Alarm_Active"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
