<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-10"/>
  <contentHeader name="Guard"/><types/><instances/>
  <pous>
    <pou name="Guard" pouType="program">
      <interface>
        <inputVars><variable name="Inhibit"><type><BOOL/></type></variable></inputVars>
        <outputVars><variable name="Hazard"><type><BOOL/></type></variable></outputVars>
      </interface>
      <body><LD>
        <rung localId="1"><contact variable="Inhibit" negated="negated"/><coil variable="Hazard"/></rung>
      </LD></body>
    </pou>
  </pous>
</project>
