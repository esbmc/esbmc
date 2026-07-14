<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-01-01"/>
  <contentHeader name="UnknownElem"/>
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
      </interface>
      <body>
        <LD>
          <rung localId="1">
            <contact variable="A"/>
            <!-- unsupported Tier-3 element -->
            <jumpLabel name="MyLabel"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
