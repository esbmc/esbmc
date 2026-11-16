<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="ESBMC" productVersion="1.0" creationDateTime="2026-10-17"/>
  <contentHeader name="IecOperatorDiagnostic"/>
  <types/>
  <instances/>
  <pous>
    <pou name="IecOperatorDiagnostic" pouType="program">
      <interface>
        <inputVars>
          <variable name="Motor_A"><type><BOOL/></type></variable>
          <variable name="Motor_B"><type><BOOL/></type></variable>
        </inputVars>
      </interface>
      <body>
        <LD>
          <rung localId="1" lineNumber="1">
            <contact variable="Motor_A"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>