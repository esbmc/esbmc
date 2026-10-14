<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-08-27"/>
  <contentHeader name="FbdBody"/>
  <types/>
  <instances/>
  <pous>
    <pou name="FbdBody" pouType="program">
      <interface>
        <outputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <FBD>
          <inVariable localId="1"><expression>TRUE</expression></inVariable>
          <outVariable localId="2">
            <connectionPointIn><connection refLocalId="1"/></connectionPointIn>
            <expression>A</expression></outVariable>
        </FBD>
      </body>
    </pou>
  </pous>
</project>
