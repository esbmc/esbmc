<?xml version="1.0" encoding="UTF-8"?>
<!-- SFC body with an LD step action: the parser can reach the action's rungs,
     but not the step sequencing that gates them (#7354). -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-08-27"/>
  <contentHeader name="Mutex"/>
  <types/>
  <instances/>
  <pous>
    <pou name="Mutex" pouType="program">
      <interface>
        <outputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <SFC>
          <step localId="1" name="Init" initialStep="true"/>
          <transition localId="2">
            <condition><connectionPointIn><connection refLocalId="1"/></connectionPointIn></condition>
          </transition>
        </SFC>
      </body>
      <actions>
        <action name="Drive">
          <body>
            <LD>
              <rung localId="1" lineNumber="1"><coil variable="A"/></rung>
              <rung localId="2" lineNumber="2"><coil variable="B"/></rung>
            </LD>
          </body>
        </action>
      </actions>
    </pou>
  </pous>
</project>
