<?xml version="1.0" encoding="UTF-8"?>
<!-- CS8: Elevator Control - SAFE version v2 -->
<!-- Fix: Door_Open now checks Motor_Up and Motor_Down directly -->
<!-- instead of Motor_Running (which had a 1-scan delay) -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-12"/>
  <contentHeader name="Elevator_Safe"/>
  <types/>
  <instances>
    <configurations>
      <configuration name="Config">
        <resource name="Resource">
          <task name="MainTask" priority="0" interval="T#10ms"/>
          <pouInstance name="Elevator_POU" typeName="Elevator_Safe"/>
        </resource>
      </configuration>
    </configurations>
  </instances>
  <pous>
    <pou name="Elevator_Safe" pouType="program">
      <interface>
        <inputVars>
          <variable name="At_Floor1"><type><BOOL/></type></variable>
          <variable name="At_Floor2"><type><BOOL/></type></variable>
          <variable name="At_Floor3"><type><BOOL/></type></variable>
          <variable name="Call_Floor1"><type><BOOL/></type></variable>
          <variable name="Call_Floor2"><type><BOOL/></type></variable>
          <variable name="Call_Floor3"><type><BOOL/></type></variable>
          <variable name="Emergency_Stop"><type><BOOL/></type></variable>
          <variable name="Door_Closed"><type><BOOL/></type></variable>
          <variable name="Overload"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="Motor_Up"><type><BOOL/></type></variable>
          <variable name="Motor_Down"><type><BOOL/></type></variable>
          <variable name="Door_Open"><type><BOOL/></type></variable>
          <variable name="Alarm_Active"><type><BOOL/></type></variable>
          <variable name="Motor_Running"><type><BOOL/></type></variable>
        </outputVars>
        <localVars>
          <variable name="At_Floor"><type><BOOL/></type></variable>
          <variable name="TON1_IN"><type><BOOL/></type></variable>
          <variable name="TON1_Q"><type><BOOL/></type></variable>
          <variable name="TON1_ET"><type><INT/></type></variable>
          <variable name="TON1_PT"><type><INT/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <!-- Rung 1-3: At_Floor composite -->
          <rung localId="1" lineNumber="1">
            <contact variable="At_Floor1"/>
            <coil variable="At_Floor"/>
          </rung>
          <rung localId="2" lineNumber="2">
            <contact variable="At_Floor2"/>
            <coil variable="At_Floor"/>
          </rung>
          <rung localId="3" lineNumber="3">
            <contact variable="At_Floor3"/>
            <coil variable="At_Floor"/>
          </rung>
          <!-- Rung 4-6: Motor_Up with full safety interlocks -->
          <rung localId="4" lineNumber="4">
            <contact variable="Call_Floor2"/>
            <contact variable="At_Floor1"/>
            <contact variable="Door_Closed"/>
            <contact variable="Emergency_Stop" negated="negated"/>
            <contact variable="Overload" negated="negated"/>
            <contact variable="Motor_Down" negated="negated"/>
            <coil variable="Motor_Up"/>
          </rung>
          <rung localId="5" lineNumber="5">
            <contact variable="Call_Floor3"/>
            <contact variable="At_Floor1"/>
            <contact variable="Door_Closed"/>
            <contact variable="Emergency_Stop" negated="negated"/>
            <contact variable="Overload" negated="negated"/>
            <contact variable="Motor_Down" negated="negated"/>
            <coil variable="Motor_Up"/>
          </rung>
          <rung localId="6" lineNumber="6">
            <contact variable="Call_Floor3"/>
            <contact variable="At_Floor2"/>
            <contact variable="Door_Closed"/>
            <contact variable="Emergency_Stop" negated="negated"/>
            <contact variable="Overload" negated="negated"/>
            <contact variable="Motor_Down" negated="negated"/>
            <coil variable="Motor_Up"/>
          </rung>
          <!-- Rung 7-9: Motor_Down with full safety interlocks -->
          <rung localId="7" lineNumber="7">
            <contact variable="Call_Floor1"/>
            <contact variable="At_Floor2"/>
            <contact variable="Door_Closed"/>
            <contact variable="Emergency_Stop" negated="negated"/>
            <contact variable="Overload" negated="negated"/>
            <contact variable="Motor_Up" negated="negated"/>
            <coil variable="Motor_Down"/>
          </rung>
          <rung localId="8" lineNumber="8">
            <contact variable="Call_Floor1"/>
            <contact variable="At_Floor3"/>
            <contact variable="Door_Closed"/>
            <contact variable="Emergency_Stop" negated="negated"/>
            <contact variable="Overload" negated="negated"/>
            <contact variable="Motor_Up" negated="negated"/>
            <coil variable="Motor_Down"/>
          </rung>
          <rung localId="9" lineNumber="9">
            <contact variable="Call_Floor2"/>
            <contact variable="At_Floor3"/>
            <contact variable="Door_Closed"/>
            <contact variable="Emergency_Stop" negated="negated"/>
            <contact variable="Overload" negated="negated"/>
            <contact variable="Motor_Up" negated="negated"/>
            <coil variable="Motor_Down"/>
          </rung>
          <!-- Rung 10-11: Motor_Running directly from Motor_Up/Down -->
          <rung localId="10" lineNumber="10">
            <contact variable="Motor_Up"/>
            <coil variable="Motor_Running"/>
          </rung>
          <rung localId="11" lineNumber="11">
            <contact variable="Motor_Down"/>
            <coil variable="Motor_Running"/>
          </rung>
          <!-- Rung 12: Door opens ONLY when at floor AND Motor_Up=0 -->
          <!-- AND Motor_Down=0 (direct check, not via Motor_Running) -->
          <rung localId="12" lineNumber="12">
            <contact variable="At_Floor"/>
            <contact variable="Motor_Up" negated="negated"/>
            <contact variable="Motor_Down" negated="negated"/>
            <contact variable="Emergency_Stop" negated="negated"/>
            <coil variable="Door_Open"/>
          </rung>
          <!-- Rung 13-14: Door timer -->
          <rung localId="13" lineNumber="13">
            <contact variable="Door_Open"/>
            <coil variable="TON1_IN"/>
          </rung>
          <rung localId="14" lineNumber="14">
            <block typeName="TON" instanceName="TON1" lineNumber="14">
              <variable formalParameter="IN">TON1_IN</variable>
              <variable formalParameter="PT">TON1_PT</variable>
              <variable formalParameter="Q">TON1_Q</variable>
              <variable formalParameter="ET">TON1_ET</variable>
            </block>
          </rung>
          <!-- Rung 15-16: Alarm -->
          <rung localId="15" lineNumber="15">
            <contact variable="Emergency_Stop"/>
            <coil variable="Alarm_Active"/>
          </rung>
          <rung localId="16" lineNumber="16">
            <contact variable="Overload"/>
            <coil variable="Alarm_Active"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
