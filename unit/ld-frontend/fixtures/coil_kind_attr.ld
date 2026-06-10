<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-01-01"/>
  <contentHeader name="CoilKindAttr"/>
  <types/>
  <instances>
    <configurations>
      <configuration name="Config">
        <resource name="Resource">
          <task name="MainTask" priority="0" interval="T#10ms"/>
          <pouInstance name="Test_POU" typeName="Test"/>
        </resource>
      </configuration>
    </configurations>
  </instances>
  <pous>
    <pou name="Test" pouType="program">
      <interface>
        <inputVars>
          <variable name="Trigger"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="Latch"><type><BOOL/></type></variable>
          <variable name="Clear"><type><BOOL/></type></variable>
          <variable name="Out"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <!-- Rung 1: set coil via kind attribute -->
          <rung localId="1">
            <contact variable="Trigger"/>
            <coil kind="set" variable="Latch"/>
          </rung>
          <!-- Rung 2: reset coil via kind attribute -->
          <rung localId="2">
            <contact variable="Trigger"/>
            <coil kind="reset" variable="Clear"/>
          </rung>
          <!-- Rung 3: output coil (no kind attribute — default) -->
          <rung localId="3">
            <contact variable="Trigger"/>
            <coil variable="Out"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
