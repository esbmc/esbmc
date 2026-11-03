<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="ESBMC" productVersion="1.0" creationDateTime="2026-10-07"/>
  <contentHeader name="Mutex3"/>
  <types/><instances/>
  <pous>
    <pou name="Mutex3" pouType="program">
      <interface>
        <inputVars>
          <variable name="a"><type><BOOL/></type></variable>
          <variable name="b"><type><BOOL/></type></variable>
          <variable name="c"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="X"><type><BOOL/></type></variable>
          <variable name="Y"><type><BOOL/></type></variable>
          <variable name="Z"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><contact variable="a"/><coil variable="X"/></rung>
          <rung localId="2"><contact variable="b"/><coil variable="Y"/></rung>
          <rung localId="3"><contact variable="c"/><contact variable="a" negated="true"/><contact variable="b" negated="true"/><coil variable="Z"/></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
