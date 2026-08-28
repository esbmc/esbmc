<?xml version="1.0" encoding="UTF-8"?>
<!-- <addData> (PLCopen's vendor extension point) and <documentation> each
     nest a <body> under the POU that is not a POU body (#7354). -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-08-27"/>
  <contentHeader name="Alias"/>
  <types/>
  <instances/>
  <pous>
    <pou name="Alias" pouType="program">
      <interface>
        <inputVars>
          <variable name="Enable"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <rung localId="1" lineNumber="1">
            <contact variable="Enable"/>
            <coil variable="A"/>
          </rung>
          <rung localId="2" lineNumber="2">
            <contact variable="Enable" negated="true"/>
            <coil variable="B"/>
          </rung>
        </LD>
      </body>
      <addData>
        <data name="vendor.widget" handleUnknown="implementation">
          <widget><body><canvas/></body></widget>
        </data>
      </addData>
      <documentation>
        <xhtml xmlns="http://www.w3.org/1999/xhtml"><body><p>Notes.</p></body></xhtml>
      </documentation>
    </pou>
  </pous>
</project>
