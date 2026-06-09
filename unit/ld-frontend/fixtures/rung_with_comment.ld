<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-01-01"/>
  <contentHeader name="RungComment"/>
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
          <variable name="A"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="B"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <rung localId="1">
            <!-- This XML comment is a non-element child and must be skipped -->
            <contact variable="A"/>
            <!-- Another comment between elements -->
            <coil variable="B"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
