<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-05"/>
  <contentHeader name="Timer"/>
  <types/><instances/>
  <pous>
    <pou name="Timer" pouType="program">
      <interface>
        <inputVars><variable name="IN1"><type><BOOL/></type></variable></inputVars>
        <localVars><variable name="Q1"><type><BOOL/></type></variable></localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><contact variable="IN1"/><coil variable="Q1"/></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
