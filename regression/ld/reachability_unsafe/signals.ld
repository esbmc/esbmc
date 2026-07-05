<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-10"/>
  <contentHeader name="Signals"/>
  <types/>
  <instances/>
  <pous>
    <pou name="Signals" pouType="program">
      <interface>
        <inputVars>
          <variable name="Alarm"><type><BOOL/></type></variable>
          <variable name="Btn"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="Shutdown"><type><BOOL/></type></variable>
          <variable name="Light"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <rung localId="1" lineNumber="1">
            <contact variable="Alarm"/>
            <coil variable="Shutdown"/>
          </rung>
          <rung localId="2" lineNumber="2">
            <contact variable="Btn"/>
            <coil variable="Light"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
