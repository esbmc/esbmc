<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-01-01"/>
  <contentHeader name="DerivedType"/>
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
        <localVars>
          <!-- derived name resolves to a known base type -->
          <variable name="Count"><type><derived name="INT"/></type></variable>
          <!-- derived name is an unknown user type; falls back to BOOL -->
          <variable name="Flag"><type><derived name="UserFBType"/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"/>
        </LD>
      </body>
    </pou>
  </pous>
</project>
