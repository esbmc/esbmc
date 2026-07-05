<?xml version="1.0" encoding="UTF-8"?>
<!-- CS2: Multi-conveyor sequential startup with TON timer confirmation -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-09"/>
  <contentHeader name="ConveyorSequencing"/>
  <types/>
  <instances>
    <configurations>
      <configuration name="Config">
        <resource name="Resource">
          <task name="MainTask" priority="0" interval="T#10ms"/>
          <pouInstance name="Conveyor_POU" typeName="ConveyorSequencing"/>
        </resource>
      </configuration>
    </configurations>
  </instances>
  <pous>
    <pou name="ConveyorSequencing" pouType="program">
      <interface>
        <inputVars>
          <variable name="Start_Button"><type><BOOL/></type></variable>
          <variable name="Stop_Button"><type><BOOL/></type></variable>
          <variable name="Emergency_Stop"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="Conveyor_Running"><type><BOOL/></type></variable>
          <variable name="Belt_1"><type><BOOL/></type></variable>
          <variable name="Belt_2"><type><BOOL/></type></variable>
        </outputVars>
        <localVars>
          <variable name="TON1_IN"><type><BOOL/></type></variable>
          <variable name="TON1_Q"><type><BOOL/></type></variable>
          <variable name="TON1_ET"><type><INT/></type></variable>
          <variable name="TON1_PT"><type><INT/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <!-- Rung 1: Start request sets TON1 input -->
          <rung localId="1" lineNumber="1">
            <contact variable="Start_Button"/>
            <contact variable="Emergency_Stop" negated="negated"/>
            <coil variable="TON1_IN"/>
          </rung>
          <!-- Rung 2: TON timer block (2-scan confirmation delay) -->
          <rung localId="2" lineNumber="2">
            <block typeName="TON" instanceName="TON1" lineNumber="2">
              <variable formalParameter="IN">TON1_IN</variable>
              <variable formalParameter="PT">TON1_PT</variable>
              <variable formalParameter="Q">TON1_Q</variable>
              <variable formalParameter="ET">TON1_ET</variable>
            </block>
          </rung>
          <!-- Rung 3: Start conveyor after timer confirms -->
          <rung localId="3" lineNumber="3">
            <contact variable="TON1_Q"/>
            <contact variable="Stop_Button" negated="negated"/>
            <coil variable="Conveyor_Running"/>
          </rung>
          <!-- Rung 4: Belt 1 follows conveyor -->
          <rung localId="4" lineNumber="4">
            <contact variable="Conveyor_Running"/>
            <coil variable="Belt_1"/>
          </rung>
          <!-- Rung 5: Belt 2 follows conveyor -->
          <rung localId="5" lineNumber="5">
            <contact variable="Conveyor_Running"/>
            <coil variable="Belt_2"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
