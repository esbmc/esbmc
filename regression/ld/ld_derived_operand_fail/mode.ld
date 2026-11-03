<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-12"/>
  <contentHeader name="P"/>
  <types/><instances/>
  <pous>
    <pou name="P" pouType="program">
      <!-- A variable of a derived type that is not a function block cannot be a coil target. -->
      <interface>
        <inputVars><variable name="a"><type><BOOL/></type></variable></inputVars>
        <localVars><variable name="m"><type><derived name="Mode"/></type></variable></localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><contact variable="a"/><coil variable="m"/></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
