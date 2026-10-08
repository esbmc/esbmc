<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="ESBMC" productVersion="1.0" creationDateTime="2026-10-06"/>
  <contentHeader name="T"/>
  <types/><instances/>
  <pous>
    <pou name="T" pouType="program">
      <interface>
        <localVars>
          <variable name="U"><type><UINT/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><contact variable="U"/><coil variable="B"/></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
