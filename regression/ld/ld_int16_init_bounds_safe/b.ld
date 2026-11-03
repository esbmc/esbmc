<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="ESBMC" productVersion="1.0" creationDateTime="2026-10-06"/>
  <contentHeader name="T"/>
  <types/><instances/>
  <pous>
    <pou name="T" pouType="program">
      <interface>
        <localVars>
          <variable name="Hi"><type><INT/></type><initialValue><simpleValue value="32767"/></initialValue></variable>
          <variable name="Lo"><type><INT/></type><initialValue><simpleValue value="-32768"/></initialValue></variable>
          <variable name="Ok"><type><BOOL/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><contact variable="Hi"/><contact variable="Lo"/><coil variable="Ok"/></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
