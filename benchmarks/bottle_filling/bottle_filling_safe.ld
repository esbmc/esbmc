<?xml version="1.0" encoding="UTF-8"?>
<!-- CS6: Bottle Filling Station - SAFE version -->
<!-- Fix 1: Fill_Valve requires Bottle_Present -->
<!-- Fix 2: Fill_Valve closes when Level_Full -->
<!-- Fix 3: Emergency_Stop forces Fill_Valve closed -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-12"/>
  <contentHeader name="BottleFilling_Safe"/>
  <types/>
  <instances>
    <configurations>
      <configuration name="Config">
        <resource name="Resource">
          <task name="MainTask" priority="0" interval="T#10ms"/>
          <pouInstance name="BottleFilling_POU" typeName="BottleFilling_Safe"/>
        </resource>
      </configuration>
    </configurations>
  </instances>
  <pous>
    <pou name="BottleFilling_Safe" pouType="program">
      <interface>
        <inputVars>
          <variable name="Start_Button"><type><BOOL/></type></variable>
          <variable name="Stop_Button"><type><BOOL/></type></variable>
          <variable name="Emergency_Stop"><type><BOOL/></type></variable>
          <variable name="Bottle_Present"><type><BOOL/></type></variable>
          <variable name="Level_Full"><type><BOOL/></type></variable>
          <variable name="Conveyor_Ready"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="Fill_Valve"><type><BOOL/></type></variable>
          <variable name="Conveyor_Motor"><type><BOOL/></type></variable>
          <variable name="Filling_Active"><type><BOOL/></type></variable>
          <variable name="Alarm_Overfill"><type><BOOL/></type></variable>
          <variable name="Alarm_No_Bottle"><type><BOOL/></type></variable>
        </outputVars>
        <localVars>
          <variable name="System_Running"><type><BOOL/></type></variable>
          <variable name="Filling_Done"><type><BOOL/></type></variable>
          <variable name="TON1_IN"><type><BOOL/></type></variable>
          <variable name="TON1_Q"><type><BOOL/></type></variable>
          <variable name="TON1_ET"><type><INT/></type></variable>
          <variable name="TON1_PT"><type><INT/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <!-- Rung 1: System running - requires Start, no Emergency, no Stop -->
          <rung localId="1" lineNumber="1">
            <contact variable="Start_Button"/>
            <contact variable="Emergency_Stop" negated="negated"/>
            <contact variable="Stop_Button" negated="negated"/>
            <coil variable="System_Running"/>
          </rung>
          <!-- Rung 2: Fill Valve - FIX: requires Bottle_Present AND not Level_Full -->
          <rung localId="2" lineNumber="2">
            <contact variable="System_Running"/>
            <contact variable="Bottle_Present"/>
            <contact variable="Level_Full" negated="negated"/>
            <contact variable="Filling_Done" negated="negated"/>
            <contact variable="Emergency_Stop" negated="negated"/>
            <coil variable="Fill_Valve"/>
          </rung>
          <!-- Rung 3: Filling timer input -->
          <rung localId="3" lineNumber="3">
            <contact variable="Fill_Valve"/>
            <coil variable="TON1_IN"/>
          </rung>
          <!-- Rung 4: TON1 timer block -->
          <rung localId="4" lineNumber="4">
            <block typeName="TON" instanceName="TON1" lineNumber="4">
              <variable formalParameter="IN">TON1_IN</variable>
              <variable formalParameter="PT">TON1_PT</variable>
              <variable formalParameter="Q">TON1_Q</variable>
              <variable formalParameter="ET">TON1_ET</variable>
            </block>
          </rung>
          <!-- Rung 5: Filling done when timer fires OR level sensor active -->
          <rung localId="5" lineNumber="5">
            <contact variable="TON1_Q"/>
            <coil variable="Filling_Done"/>
          </rung>
          <!-- Rung 6: Level_Full also sets Filling_Done immediately -->
          <rung localId="6" lineNumber="6">
            <contact variable="Level_Full"/>
            <coil variable="Filling_Done"/>
          </rung>
          <!-- Rung 7: Filling active indicator -->
          <rung localId="7" lineNumber="7">
            <contact variable="Fill_Valve"/>
            <coil variable="Filling_Active"/>
          </rung>
          <!-- Rung 8: Conveyor advances when filling done, no emergency, conveyor ready -->
          <rung localId="8" lineNumber="8">
            <contact variable="Filling_Done"/>
            <contact variable="Emergency_Stop" negated="negated"/>
            <contact variable="Conveyor_Ready"/>
            <coil variable="Conveyor_Motor"/>
          </rung>
          <!-- Rung 9: Overfill alarm - valve open AND level full (should not happen with fix) -->
          <rung localId="9" lineNumber="9">
            <contact variable="Level_Full"/>
            <contact variable="Fill_Valve"/>
            <coil variable="Alarm_Overfill"/>
          </rung>
          <!-- Rung 10: No-bottle alarm -->
          <rung localId="10" lineNumber="10">
            <contact variable="Fill_Valve"/>
            <contact variable="Bottle_Present" negated="negated"/>
            <coil variable="Alarm_No_Bottle"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
