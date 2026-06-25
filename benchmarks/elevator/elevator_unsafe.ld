<?xml version="1.0" encoding="UTF-8"?>
<!-- CS8: Elevator Control - 3 floors - UNSAFE version -->
<!-- Industrial elevator controller inspired by FOSSEE/OpenPLC examples -->
<!-- and IEC 61131-3 educational case studies (Parr, 2003) -->
<!-- Bugs: Door can open between floors; motor can reverse without stopping -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-12"/>
  <contentHeader name="Elevator_Unsafe"/>
  <types/>
  <instances>
    <configurations>
      <configuration name="Config">
        <resource name="Resource">
          <task name="MainTask" priority="0" interval="T#10ms"/>
          <pouInstance name="Elevator_POU" typeName="Elevator_Unsafe"/>
        </resource>
      </configuration>
    </configurations>
  </instances>
  <pous>
    <pou name="Elevator_Unsafe" pouType="program">
      <interface>
        <inputVars>
          <!-- Floor sensors: TRUE when cabin is at that floor -->
          <variable name="At_Floor1"><type><BOOL/></type></variable>
          <variable name="At_Floor2"><type><BOOL/></type></variable>
          <variable name="At_Floor3"><type><BOOL/></type></variable>
          <!-- Call buttons from each floor -->
          <variable name="Call_Floor1"><type><BOOL/></type></variable>
          <variable name="Call_Floor2"><type><BOOL/></type></variable>
          <variable name="Call_Floor3"><type><BOOL/></type></variable>
          <!-- Safety inputs -->
          <variable name="Emergency_Stop"><type><BOOL/></type></variable>
          <variable name="Door_Closed"><type><BOOL/></type></variable>
          <variable name="Overload"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="Motor_Up"><type><BOOL/></type></variable>
          <variable name="Motor_Down"><type><BOOL/></type></variable>
          <variable name="Door_Open"><type><BOOL/></type></variable>
          <variable name="Alarm_Active"><type><BOOL/></type></variable>
        </outputVars>
        <localVars>
          <variable name="At_Floor"><type><BOOL/></type></variable>
          <variable name="Going_Up"><type><BOOL/></type></variable>
          <variable name="Going_Down"><type><BOOL/></type></variable>
          <variable name="TON1_IN"><type><BOOL/></type></variable>
          <variable name="TON1_Q"><type><BOOL/></type></variable>
          <variable name="TON1_ET"><type><INT/></type></variable>
          <variable name="TON1_PT"><type><INT/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <!-- Rung 1: Cabin is at a floor when any floor sensor active -->
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
          <!-- Rung 4: Motor Up - go up when call from higher floor -->
          <!-- BUG: no check that door is closed, no emergency check -->
          <rung localId="4" lineNumber="4">
            <contact variable="Call_Floor2"/>
            <contact variable="At_Floor1"/>
            <coil variable="Motor_Up"/>
          </rung>
          <rung localId="5" lineNumber="5">
            <contact variable="Call_Floor3"/>
            <contact variable="At_Floor1"/>
            <coil variable="Motor_Up"/>
          </rung>
          <rung localId="6" lineNumber="6">
            <contact variable="Call_Floor3"/>
            <contact variable="At_Floor2"/>
            <coil variable="Motor_Up"/>
          </rung>
          <!-- Rung 7: Motor Down - go down when call from lower floor -->
          <!-- BUG: no check that door is closed, no emergency check -->
          <rung localId="7" lineNumber="7">
            <contact variable="Call_Floor1"/>
            <contact variable="At_Floor2"/>
            <coil variable="Motor_Down"/>
          </rung>
          <rung localId="8" lineNumber="8">
            <contact variable="Call_Floor1"/>
            <contact variable="At_Floor3"/>
            <coil variable="Motor_Down"/>
          </rung>
          <rung localId="9" lineNumber="9">
            <contact variable="Call_Floor2"/>
            <contact variable="At_Floor3"/>
            <coil variable="Motor_Down"/>
          </rung>
          <!-- Rung 10: Door opens when at a floor -->
          <!-- BUG: no check that motor is stopped -->
          <rung localId="10" lineNumber="10">
            <contact variable="At_Floor"/>
            <coil variable="Door_Open"/>
          </rung>
          <!-- Rung 11: Door timer -->
          <rung localId="11" lineNumber="11">
            <contact variable="Door_Open"/>
            <coil variable="TON1_IN"/>
          </rung>
          <!-- Rung 12: TON1 timer -->
          <rung localId="12" lineNumber="12">
            <block typeName="TON" instanceName="TON1" lineNumber="12">
              <variable formalParameter="IN">TON1_IN</variable>
              <variable formalParameter="PT">TON1_PT</variable>
              <variable formalParameter="Q">TON1_Q</variable>
              <variable formalParameter="ET">TON1_ET</variable>
            </block>
          </rung>
          <!-- Rung 13: Alarm on emergency or overload -->
          <rung localId="13" lineNumber="13">
            <contact variable="Emergency_Stop"/>
            <coil variable="Alarm_Active"/>
          </rung>
          <rung localId="14" lineNumber="14">
            <contact variable="Overload"/>
            <coil variable="Alarm_Active"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
