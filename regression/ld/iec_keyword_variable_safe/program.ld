<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="ESBMC" productVersion="1.0" creationDateTime="2026-10-17"/>
  <contentHeader name="IecKeywordVariableSafe"/>
  <types/>
  <instances/>
  <pous>
    <pou name="IecKeywordVariableSafe" pouType="program">
      <interface>
        <inputVars>
          <variable name="Xor"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="Motor_A"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <rung localId="1" lineNumber="1">
            <contact variable="Xor"/>
            <coil variable="Motor_A"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
