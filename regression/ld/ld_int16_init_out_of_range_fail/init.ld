<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="ESBMC" productVersion="1.0" creationDateTime="2026-10-06"/>
  <contentHeader name="Init"/>
  <types/><instances/>
  <pous>
    <pou name="Init" pouType="program">
      <interface>
        <localVars>
          <variable name="Big"><type><INT/></type><initialValue><simpleValue value="40000"/></initialValue></variable>
          <variable name="Y"><type><BOOL/></type></variable>
        </localVars>
      </interface>
      <body><LD><rung localId="1"><contact variable="Big"/><coil variable="Y"/></rung></LD></body>
    </pou>
  </pous>
</project>
